use derive_more::{Add, Sum};
use std::simd::{i16x8, i64x8};
use timing_shield::{TpBool, TpCondSwap};
use tp_fixedpoint::TpFixed64;

#[derive(Clone, Copy, Sum, Add)]
pub struct TpFixed64x8<const F: usize>(pub i64x8);

impl<const F: usize> TpFixed64x8<F> {
    pub fn splat(e: TpFixed64<F>) -> Self {
        Self(i64x8::splat(e.into_inner().expose()))
    }
}


impl<const F: usize> std::ops::Mul for TpFixed64x8<F> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self((self.0 * rhs.0) >> F as i64)
    }
}

impl<const F: usize> std::ops::MulAssign for TpFixed64x8<F> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
        self.0 >>= F as i64;
    }
}

impl<const F: usize> TpCondSwap for TpFixed64x8<F> {
    #[inline(always)]
    fn tp_cond_swap(condition: TpBool, a: &mut Self, b: &mut Self) {
        // Zero-extend condition to this type's width
        let cond_zx = condition.as_i64().expose();

        // Create mask of 11...11 for true or 00...00 for false
        let mask = i64x8::splat(!(cond_zx - 1));

        // swapper will be a XOR b for true or 00...00 for false
        let swapper = (a.0 ^ b.0) & mask;

        a.0 ^= swapper;
        b.0 ^= swapper;
    }
}

//#[derive(Clone)]
//pub struct DArray8x1<const F: usize> {
//pub a: Array8x1<F>,
//pub e: i16x8,
//}

//impl<const F: usize> std::ops::MulAssign<Array8x1<F>> for DArray8x1<F> {
//fn mul_assign(&mut self, rhs: Array8x1<F>) {
//self.a *= rhs;
//}
//}

////impl<const F: usize> DArray8x1<F> {
////pub fn assign_mul_a1(&mut self, a: &Array8x1) {
////self.array *= a;
////self.array >>= F as u64;
////}
////}

//pub struct DArray8x2<const F: usize> {
//pub a: Vec<Array8x1<F>>,
//pub e: i16x8,
//}

//impl<const F: usize> DArray8x2<F> {
//pub fn assign(&mut self, rhs: &Self)  {
//self.a.clone_from_slice(rhs.a.as_slice());
//self.e = rhs.e;
//}
//pub fn sum_by_row(&self) -> DArray8x1<F> {
//DArray8x1 {
//a: self.a.iter().cloned().sum(),
//e: self.e,
//}
//}
//}

//impl<const F: usize> std::ops::Add<DArray8x1<F>> for DArray8x2<F> {
//type Output = Self;
//fn add(self, rhs: DArray8x1<F>) -> Self::Output {
//Self {
//a: self.a.into_iter().map(|row| row + rhs.a).collect(),
//e: self.e,
//}
//}
//}
