use crate::chess::{bitboard::Bitboard, coords::Square};

pub struct HyperbolaQsc {
    mask: [Mask; 64],
    rank_attack: [u8; 512],
}

#[derive(Clone, Copy)]
struct Mask {
    diagonal: u64,
    antidiagonal: u64,
    vertical: u64,
}

impl HyperbolaQsc {
    pub const fn new() -> Self {
        let mask = Self::init_mask();
        let rank_attack = Self::init_rank();

        Self { mask, rank_attack }
    }

    const fn init_mask() -> [Mask; 64] {
        let mut mask = [Mask {
            diagonal: 0,
            antidiagonal: 0,
            vertical: 0,
        }; 64];

        let mut x = 0;
        while x < 64 {
            let mut d = [0i32; 64];

            // Calculate directions
            let mut i = -1;
            while i <= 1 {
                let mut j = -1;
                while j <= 1 {
                    if i == 0 && j == 0 {
                        j += 1;
                        continue;
                    }

                    let mut f = (x & 7) as i32;
                    let mut r = (x >> 3) as i32;

                    r += i;
                    f += j;
                    while r >= 0 && r < 8 && f >= 0 && f < 8 {
                        let y = 8 * r + f;
                        d[y as usize] = 8 * i + j;
                        r += i;
                        f += j;
                    }
                    j += 1;
                }
                i += 1;
            }

            // Generate masks
            let mut y = x as i32 - 9;
            while y >= 0 && d[y as usize] == -9 {
                mask[x].diagonal |= 1u64 << y;
                y -= 9;
            }

            let mut y = x as i32 + 9;
            while y < 64 && d[y as usize] == 9 {
                mask[x].diagonal |= 1u64 << y;
                y += 9;
            }

            let mut y = x as i32 - 7;
            while y >= 0 && d[y as usize] == -7 {
                mask[x].antidiagonal |= 1u64 << y;
                y -= 7;
            }

            let mut y = x as i32 + 7;
            while y < 64 && d[y as usize] == 7 {
                mask[x].antidiagonal |= 1u64 << y;
                y += 7;
            }

            let mut y = x as i32 - 8;
            while y >= 0 {
                mask[x].vertical |= 1u64 << y;
                y -= 8;
            }

            let mut y = x as i32 + 8;
            while y < 64 {
                mask[x].vertical |= 1u64 << y;
                y += 8;
            }

            x += 1;
        }
        mask
    }

    const fn init_rank() -> [u8; 512] {
        let mut rank_attack = [0u8; 512];

        let mut x = 0;
        while x < 64 {
            let mut f = 0;
            while f < 8 {
                let o = 2 * x;
                let mut y2 = 0u8;

                // Left side
                let mut x2 = f as i32 - 1;
                while x2 >= 0 {
                    let b = 1 << x2;
                    y2 |= b as u8;
                    if (o & b) == b {
                        break;
                    }
                    x2 -= 1;
                }

                // Right side
                let mut x2 = f + 1;
                while x2 < 8 {
                    let b = 1 << x2;
                    y2 |= b as u8;
                    if (o & b) == b {
                        break;
                    }
                    x2 += 1;
                }

                rank_attack[x * 8 + f] = y2;
                f += 1;
            }
            x += 1;
        }
        rank_attack
    }

    #[inline]
    const fn bit_bswap(b: u64) -> u64 {
        b.swap_bytes()
    }

    fn attack(pieces: u64, x: u32, mask: u64) -> u64 {
        let o = pieces & mask;
        ((o.wrapping_sub(1u64 << x))
            ^ Self::bit_bswap(Self::bit_bswap(o).wrapping_sub(0x8000000000000000u64 >> x)))
            & mask
    }

    pub fn horizontal_attack(&self, pieces: u64, x: u32) -> u64 {
        let file_mask = x & 7;
        let rank_mask = x & 56;
        let o = (pieces >> rank_mask) & 126;

        let idx = o * 4 + file_mask as u64;

        (self.rank_attack[idx as usize] as u64) << rank_mask
    }

    pub fn vertical_attack(&self, pieces: u64, sq: u32) -> u64 {
        Self::attack(pieces, sq, self.mask[sq as usize].vertical)
    }

    pub fn diagonal_attack(&self, pieces: u64, sq: u32) -> u64 {
        Self::attack(pieces, sq, self.mask[sq as usize].diagonal)
    }

    pub fn antidiagonal_attack(&self, pieces: u64, sq: u32) -> u64 {
        Self::attack(pieces, sq, self.mask[sq as usize].antidiagonal)
    }

    pub fn bishop_attack(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        let sq_idx = sq.index();
        Bitboard::from_u64(
            self.diagonal_attack(occupied.bits(), sq_idx)
                | self.antidiagonal_attack(occupied.bits(), sq_idx),
        )
    }

    pub fn rook_attack(&self, sq: Square, occupied: Bitboard) -> Bitboard {
        let sq_idx = sq.index();
        Bitboard::from_u64(
            self.vertical_attack(occupied.bits(), sq_idx)
                | self.horizontal_attack(occupied.bits(), sq_idx),
        )
    }

    // pub fn queen_attack(&self, sq: Square, occupied: Bitboard) -> Bitboard {
    //     Bitboard::from_u64(
    //         self.bishop_attack(sq, occupied).bits() | self.rook_attack(sq, occupied).bits(),
    //     )
    // }
}
