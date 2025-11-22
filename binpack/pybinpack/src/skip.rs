use rand::Rng;
use sfbinpack::{
    chess::{
        color::Color, coords::Square, piece::Piece, piecetype::PieceType, position::Position,
        r#move::MoveType,
    },
    TrainingDataEntry,
};

const VALUE_NONE: i16 = 32002;
const MAX_SKIPPING_RATE: f64 = 10.0;
const DESIRED_PIECE_COUNT_WEIGHTS: [f64; 33] = [
    1.000000, 1.121094, 1.234375, 1.339844, 1.437500, 1.527344, 1.609375, 1.683594, 1.750000,
    1.808594, 1.859375, 1.902344, 1.937500, 1.964844, 1.984375, 1.996094, 2.000000, 1.996094,
    1.984375, 1.964844, 1.937500, 1.902344, 1.859375, 1.808594, 1.750000, 1.683594, 1.609375,
    1.527344, 1.437500, 1.339844, 1.234375, 1.121094, 1.000000,
];
fn sum_weights(weights: &[f64; 33]) -> f64 {
    let mut idx = 0;
    let mut acc = 0.0;
    while idx < 33 {
        acc += weights[idx];
        idx += 1;
    }
    acc
}

// DESIRED_TOTAL is computed at runtime to avoid using floating point arithmetic
// in a const function, which is unstable/unsupported on some Rust versions.

#[derive(Debug, Clone)]
pub struct SkipConfig {
    pub filtered: bool,
    pub random_fen_skipping: i32,
    pub wld_filtered: bool,
    pub early_fen_skipping: i32,
    pub simple_eval_skipping: i32,
    pub param_index: i32,
}

impl Default for SkipConfig {
    fn default() -> Self {
        Self {
            filtered: false,
            random_fen_skipping: 0,
            wld_filtered: false,
            early_fen_skipping: -1,
            simple_eval_skipping: -1,
            param_index: 0,
        }
    }
}

impl SkipConfig {
    pub fn is_active(&self) -> bool {
        self.filtered
            || self.random_fen_skipping > 0
            || self.wld_filtered
            || self.early_fen_skipping >= 0
            || self.simple_eval_skipping > 0
    }
}

pub struct SkipState {
    config: SkipConfig,
    piece_count_history_all: [f64; 33],
    piece_count_history_passed: [f64; 33],
    piece_count_history_all_total: f64,
    piece_count_history_passed_total: f64,
    alpha: f64,
    desired_total: f64,
    random_skip_probability: f64,
}

impl SkipState {
    pub fn maybe_new(config: SkipConfig) -> Option<Self> {
        if config.is_active() {
            Some(Self::new(config))
        } else {
            None
        }
    }

    fn new(config: SkipConfig) -> Self {
        let random_skip_probability = if config.random_fen_skipping > 0 {
            let denom = config.random_fen_skipping as f64 + 1.0;
            (config.random_fen_skipping as f64) / denom
        } else {
            0.0
        };

        let desired_total = sum_weights(&DESIRED_PIECE_COUNT_WEIGHTS);

        Self {
            config,
            piece_count_history_all: [0.0; 33],
            piece_count_history_passed: [0.0; 33],
            piece_count_history_all_total: 0.0,
            piece_count_history_passed_total: 0.0,
            alpha: 1.0,
            desired_total,
            random_skip_probability,
        }
    }

    pub fn should_keep(&mut self, entry: &TrainingDataEntry) -> bool {
        if !self.config.is_active() {
            return true;
        }

        let mut rng = rand::thread_rng();

        if entry.score == VALUE_NONE {
            return false;
        }

        if self.config.early_fen_skipping >= 0
            && (entry.ply as i32) <= self.config.early_fen_skipping
        {
            return false;
        }

        if self.config.random_fen_skipping > 0 && rng.gen_bool(self.random_skip_probability) {
            return false;
        }

        if self.config.filtered && (is_capturing_move(entry) || is_in_check(entry)) {
            return false;
        }

        if self.config.wld_filtered {
            let prob = (1.0 - score_result_prob(entry)).clamp(0.0, 1.0);
            if rng.gen_bool(prob) {
                return false;
            }
        }

        if self.config.simple_eval_skipping > 0 {
            let eval = simple_eval(&entry.pos).abs();
            if eval < self.config.simple_eval_skipping {
                return false;
            }
        }

        let piece_count = usize::min(entry.pos.occupied().count() as usize, 32);
        self.apply_piece_distribution(piece_count, &mut rng)
    }

    fn apply_piece_distribution(&mut self, piece_count: usize, rng: &mut impl Rng) -> bool {
        self.piece_count_history_all[piece_count] += 1.0;
        self.piece_count_history_all_total += 1.0;

        if (self.piece_count_history_all_total as u64) % 10000 == 0 {
            let mut pass = self.piece_count_history_all_total * self.desired_total;
            for (idx, weight) in DESIRED_PIECE_COUNT_WEIGHTS.iter().enumerate() {
                if *weight <= 0.0 {
                    continue;
                }
                let count = self.piece_count_history_all[idx];
                if count <= 0.0 {
                    continue;
                }
                let tmp = self.piece_count_history_all_total * weight / (self.desired_total * count);
                if tmp < pass {
                    pass = tmp;
                }
            }
            self.alpha = 1.0 / (pass * MAX_SKIPPING_RATE).max(1e-9);
        }

        let denom = self.piece_count_history_all[piece_count].max(1.0);
        let mut tmp = self.alpha
            * self.piece_count_history_all_total
            * DESIRED_PIECE_COUNT_WEIGHTS[piece_count]
            / (self.desired_total * denom);
        tmp = tmp.clamp(0.0, 1.0);
        let skip_prob = (1.0 - tmp).clamp(0.0, 1.0);
        if rng.gen_bool(skip_prob) {
            return false;
        }

        self.piece_count_history_passed[piece_count] += 1.0;
        self.piece_count_history_passed_total += 1.0;
        true
    }
}

fn is_capturing_move(entry: &TrainingDataEntry) -> bool {
    let mv = entry.mv;
    if mv.mtype() == MoveType::EnPassant {
        return true;
    }

    let from_piece = entry.pos.piece_at(mv.from());
    let to_piece = entry.pos.piece_at(mv.to());
    to_piece != Piece::none() && to_piece.color() != from_piece.color()
}

fn is_in_check(entry: &TrainingDataEntry) -> bool {
    let side = entry.pos.side_to_move();
    entry.pos.is_checked(side)
}

fn simple_eval(pos: &Position) -> i32 {
    let mut score = 0i32;
    for idx in 0..64u32 {
        let square = Square::new(idx);
        let piece = pos.piece_at(square);
        if piece == Piece::none() {
            continue;
        }

        let value = match piece.piece_type() {
            PieceType::Pawn => 100,
            PieceType::Knight => 320,
            PieceType::Bishop => 330,
            PieceType::Rook => 500,
            PieceType::Queen => 900,
            PieceType::King | PieceType::None => 0,
        };

        if piece.color() == Color::White {
            score += value;
        } else {
            score -= value;
        }
    }

    score
}

fn score_result_prob(entry: &TrainingDataEntry) -> f64 {
    let ply = (entry.ply.min(240) as f64) / 64.0;
    let as_coeffs = [-3.683_893_04, 30.070_659_21, -60.528_787_23, 149.533_785_57];
    let bs_coeffs = [-2.018_185_7, 15.856_850_38, -29.834_520_23, 47.590_788_27];

    let a = ((as_coeffs[0] * ply + as_coeffs[1]) * ply + as_coeffs[2]) * ply + as_coeffs[3];
    let mut b = ((bs_coeffs[0] * ply + bs_coeffs[1]) * ply + bs_coeffs[2]) * ply + bs_coeffs[3];
    b *= 1.5;
    if b.abs() < 1e-9 {
        b = 1e-9;
    }

    let x = ((entry.score as f64) * 100.0 / 208.0).clamp(-2000.0, 2000.0);
    let w = 1.0 / (1.0 + ((a - x) / b).exp());
    let l = 1.0 / (1.0 + ((a + x) / b).exp());
    let d = 1.0 - w - l;

    if entry.result > 0 {
        w
    } else if entry.result < 0 {
        l
    } else {
        d
    }
}
