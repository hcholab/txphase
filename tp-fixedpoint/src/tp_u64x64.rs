use derive_more::*;
use std::simd::u64x8;
use timing_shield::{TpBool, TpCondSwap, TpU64};

#[derive(From, BitXor, BitXorAssign, BitAnd, Not, Clone, Copy)]
pub struct TpU64x8(u64x8);

impl TpU64x8 {
    pub const ZERO: Self = Self(u64x8::from_array([0; 8]));

    #[inline(always)]
    pub const fn protect(v: u64x8) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub const fn expose(self) -> u64x8 {
        self.0
    }

    pub const fn as_array(&self) -> &[TpU64; 8] {
        unsafe { std::mem::transmute(self.0.as_array()) }
    }
    pub const fn to_array(self) -> [TpU64; 8] {
        unsafe { std::mem::transmute(self.0.to_array()) }
    }
}

impl TpCondSwap for TpU64x8 {
    fn tp_cond_swap(condition: TpBool, a: &mut Self, b: &mut Self) {
        // Zero-extend condition to this type's width
        let cond_zx = condition.as_u64();

        // Create mask of 11...11 for true or 00...00 for false
        let mask = Self::protect(u64x8::splat((!(cond_zx - 1)).expose()));

        // swapper will be a XOR b for true or 00...00 for false
        let swapper = (*a ^ *b) & mask;

        *a ^= swapper;
        *b ^= swapper;
    }
}
