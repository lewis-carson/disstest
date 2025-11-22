use numpy::{ndarray::Array2, IntoPyArray, PyArray1};
use pyo3::{prelude::*, types::PyTuple};
use sfbinpack::{
    chess::{color::Color, coords::Square, piece::Piece, piecetype::PieceType},
    TrainingDataEntry,
};

use crate::error::LoaderError;

#[derive(Clone, Copy)]
pub enum FeatureSet {
    HalfKP,
}

impl FeatureSet {
    pub fn try_from_name(name: &str) -> Result<Self, LoaderError> {
        match name {
            "HalfKP" => Ok(FeatureSet::HalfKP),
            other => Err(LoaderError::UnsupportedFeatureSet(other.to_string())),
        }
    }

    pub fn max_active_features(&self) -> usize {
        match self {
            FeatureSet::HalfKP => HalfKPSparse::MAX_ACTIVE_FEATURES,
        }
    }
}

pub struct SparseBatchData {
    size: usize,
    max_active_features: usize,
    is_white: Vec<f32>,
    outcome: Vec<f32>,
    score: Vec<f32>,
    white_indices: Vec<i32>,
    white_values: Vec<f32>,
    black_indices: Vec<i32>,
    black_values: Vec<f32>,
    psqt_indices: Vec<i32>,
    layer_stack_indices: Vec<i32>,
}

impl SparseBatchData {
    pub fn from_entries(entries: Vec<TrainingDataEntry>, feature_set: FeatureSet) -> Self {
        let size = entries.len();
        let max_active_features = feature_set.max_active_features();

        let mut is_white = vec![0f32; size];
        let mut outcome = vec![0f32; size];
        let mut score = vec![0f32; size];
        let mut psqt_indices = vec![0i32; size];
        let mut layer_stack_indices = vec![0i32; size];

        let mut white_indices = vec![-1i32; size * max_active_features];
        let mut white_values = vec![0f32; size * max_active_features];
        let mut black_indices = vec![-1i32; size * max_active_features];
        let mut black_values = vec![0f32; size * max_active_features];

        for (i, entry) in entries.iter().enumerate() {
            let pos = entry.pos;
            let is_white_turn = (pos.side_to_move() == Color::White) as u8;
            is_white[i] = is_white_turn as f32;
            outcome[i] = (entry.result as f32 + 1.0) * 0.5;
            score[i] = entry.score as f32;

            let piece_count = pos.occupied().count() as i32;
            let bucket = ((piece_count - 1).max(0) / 4) as i32;
            psqt_indices[i] = bucket;
            layer_stack_indices[i] = bucket;

            let offset = i * max_active_features;
            let white_slice = &mut white_indices[offset..offset + max_active_features];
            let white_values_slice = &mut white_values[offset..offset + max_active_features];
            white_slice.fill(-1);
            white_values_slice.fill(0.0);

            let black_slice = &mut black_indices[offset..offset + max_active_features];
            let black_values_slice = &mut black_values[offset..offset + max_active_features];
            black_slice.fill(-1);
            black_values_slice.fill(0.0);

            match feature_set {
                FeatureSet::HalfKP => {
                    HalfKPSparse::fill_features(
                        entry,
                        Color::White,
                        white_slice,
                        white_values_slice,
                    );
                    HalfKPSparse::fill_features(
                        entry,
                        Color::Black,
                        black_slice,
                        black_values_slice,
                    );
                }
            }
        }

        Self {
            size,
            max_active_features,
            is_white,
            outcome,
            score,
            white_indices,
            white_values,
            black_indices,
            black_values,
            psqt_indices,
            layer_stack_indices,
        }
    }

    pub fn into_py(self, py: Python<'_>) -> PyResult<PyObject> {
        let SparseBatchData {
            size,
            max_active_features,
            is_white,
            outcome,
            score,
            white_indices,
            white_values,
            black_indices,
            black_values,
            psqt_indices,
            layer_stack_indices,
        } = self;

        let them: Vec<f32> = is_white.iter().map(|v| 1.0 - *v).collect();

        let us_tensor = Array2::from_shape_vec((size, 1), is_white)
            .expect("invalid us tensor shape")
            .into_pyarray(py);
        let them_tensor = Array2::from_shape_vec((size, 1), them)
            .expect("invalid them tensor shape")
            .into_pyarray(py);
        let white_idx_tensor = Array2::from_shape_vec((size, max_active_features), white_indices)
            .expect("invalid white index shape")
            .into_pyarray(py);
        let white_val_tensor = Array2::from_shape_vec((size, max_active_features), white_values)
            .expect("invalid white values shape")
            .into_pyarray(py);
        let black_idx_tensor = Array2::from_shape_vec((size, max_active_features), black_indices)
            .expect("invalid black index shape")
            .into_pyarray(py);
        let black_val_tensor = Array2::from_shape_vec((size, max_active_features), black_values)
            .expect("invalid black values shape")
            .into_pyarray(py);
        let outcome_tensor = Array2::from_shape_vec((size, 1), outcome)
            .expect("invalid outcome shape")
            .into_pyarray(py);
        let score_tensor = Array2::from_shape_vec((size, 1), score)
            .expect("invalid score shape")
            .into_pyarray(py);
        let psqt_tensor = PyArray1::from_vec(py, psqt_indices);
        let layer_stack_tensor = PyArray1::from_vec(py, layer_stack_indices);

        let tuple = PyTuple::new(
            py,
            [
                us_tensor.to_object(py),
                them_tensor.to_object(py),
                white_idx_tensor.to_object(py),
                white_val_tensor.to_object(py),
                black_idx_tensor.to_object(py),
                black_val_tensor.to_object(py),
                outcome_tensor.to_object(py),
                score_tensor.to_object(py),
                psqt_tensor.to_object(py),
                layer_stack_tensor.to_object(py),
            ],
        );

        Ok(tuple.into())
    }
}

struct HalfKPSparse;

impl HalfKPSparse {
    pub const MAX_ACTIVE_FEATURES: usize = 32;

    fn fill_features(
        entry: &TrainingDataEntry,
        color: Color,
        indices: &mut [i32],
        values: &mut [f32],
    ) {
        let pos = entry.pos;
        let king_sq = pos.king_sq(color);
        let king_bucket = Self::orient_square(color, king_sq);
        let mut pieces = pos.occupied().bits() & !pos.pieces_bb_type(PieceType::King).bits();
        let mut count = 0usize;

        while pieces != 0 && count < indices.len() {
            let sq_idx = pieces.trailing_zeros() as u32;
            pieces &= pieces - 1;
            let square = Square::new(sq_idx);
            let piece = pos.piece_at(square);
            if piece == Piece::none() {
                continue;
            }

            let piece_type_idx = match piece.piece_type() {
                PieceType::Pawn => 0,
                PieceType::Knight => 1,
                PieceType::Bishop => 2,
                PieceType::Rook => 3,
                PieceType::Queen => 4,
                _ => continue,
            };

            let is_enemy = usize::from(piece.color() != color);
            let square_idx = Self::orient_square(color, square);

            let feature = king_bucket * 640
                + is_enemy * 320
                + piece_type_idx * 64
                + square_idx;

            indices[count] = feature as i32;
            values[count] = 1.0;
            count += 1;
        }
    }

    fn orient_square(color: Color, square: Square) -> usize {
        if color == Color::White {
            square.index() as usize
        } else {
            (square.index() ^ 56) as usize
        }
    }
}
