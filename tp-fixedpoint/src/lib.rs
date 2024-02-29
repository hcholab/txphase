#![feature(const_fn_floating_point_arithmetic)]
#![feature(const_mut_refs)]
#![feature(portable_simd)]
#![allow(dead_code)]

mod fixed_32;
mod fixed_64;
mod ln_fixed;
mod tp_i128;
mod tp_u128;
mod tp_u64x64;
pub use fixed_32::*;
pub use fixed_64::*;
pub use ln_fixed::*;
pub use num_traits;
pub use timing_shield;
pub use tp_i128::*;
pub use tp_u128::*;
pub use tp_u64x64::*;

pub trait Dot<Rhs> {
    type Output;
    fn dot(&self, rhs: &Rhs) -> Self::Output;
}
