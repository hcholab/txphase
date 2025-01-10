use derive_more::*;
use timing_shield::{TpBool, TpCondSwap};

macro_rules! tp_simd_impl {
    ($type:ty, $size:expr) => {
        paste::paste! {
            pub use [<tp_ $type x $size _inner>]::[<Tp $type:upper x $size>];
            mod [<tp_ $type x $size _inner>] {
                use super::*;
                use std::simd::[<$type x $size>];
                use timing_shield::[<Tp $type:upper>];

                #[derive(From, BitXor, BitXorAssign, BitAnd, Not, Clone, Copy, Add, AddAssign)]
                pub struct [<Tp $type:upper x $size>]([<$type x $size>]);

                impl [<Tp $type:upper x $size>] {
                    pub const ZERO: Self = Self([<$type x $size>]::from_array([0; $size]));

                    #[inline(always)]
                    pub const fn protect(v: [<$type x $size>]) -> Self {
                        Self(v)
                    }

                    #[inline(always)]
                    pub const fn expose(self) -> [<$type x $size>] {
                        self.0
                    }

                    pub const fn as_array(&self) -> &[[< Tp $type:upper >]; $size] {
                        unsafe { std::mem::transmute(self.0.as_array()) }
                    }

                    pub fn as_mut_array(&mut self) -> &mut [[< Tp $type:upper >]; $size] {
                        unsafe { std::mem::transmute(self.0.as_mut_array()) }
                    }

                    pub const fn to_array(self) -> [[< Tp $type:upper >]; $size] {
                        unsafe { std::mem::transmute(self.0.to_array()) }
                    }
                }

                impl TpCondSwap for [<Tp $type:upper x $size>] {
                    #[inline(always)]
                    fn tp_cond_swap(condition: TpBool, a: &mut Self, b: &mut Self) {
                        // Zero-extend condition to this type's width
                        let cond_zx = condition.[< as_ $type >]();

                        // Create mask of 11...11 for true or 00...00 for false
                        let mask = Self::protect([<$type x $size>]::splat((!(cond_zx - 1)).expose()));

                        // swapper will be a XOR b for true or 00...00 for false
                        let swapper = (*a ^ *b) & mask;

                        *a ^= swapper;
                        *b ^= swapper;
                    }
                }

            }


        }
    };
}

tp_simd_impl!(u64, 2);
tp_simd_impl!(u64, 4);
tp_simd_impl!(u64, 8);
