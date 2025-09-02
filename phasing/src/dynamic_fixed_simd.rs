use ndarray::Array1;
use std::mem::transmute;
use std::simd::i64x8;
use std::simd::num::{SimdInt, SimdUint};
use timing_shield::{TpI16, TpI64, TpOrd};
use tp_fixedpoint::{TpFixed64, TpU64x8};
use crate::dynamic_fixed;

pub fn sum_scale_by_column<const F: usize>(
    mut probs: Vec<i64x8>,
    probs_e: i64x8,
) -> (Array1<TpFixed64<F>>, TpI16) {
    let sum_e = probs_e.reduce_max();

    let sum_e_ = i64x8::splat(sum_e);
    let increase_e = sum_e_ - probs_e;
    increase_scale(increase_e, &mut probs);

    let mut sum = unsafe {
        Array1::from_vec(
            probs
                .into_iter()
                .map(|v| transmute(v.reduce_sum()))
                .collect::<Vec<_>>(),
        )
    };
    let mut sum_e = TpI64::protect(sum_e).as_i16();
    dynamic_fixed::renorm_scale_row(sum.view_mut(), &mut sum_e);
    (sum, sum_e)
}

pub fn increase_scale(increase_e: i64x8, probs: &mut [i64x8]) {
    for p in probs {
        *p >>= increase_e;
    }
}

//pub fn renorm_scale_row<const F: usize>(prob: &mut i64x8, prob_e: &mut i64) {
//let min_leading_zeros = prob.leading_zeros().reduce_min();
//let adjust_e = 64i64 - min_leading_zeros as i64 - F as i64;
//adjust_scale_row::<F>(adjust_e, prob, prob_e);

//}

//pub fn adjust_scale_row<const F: usize>(adjust_e: i64, prob: &mut i64x8, prob_e: &mut i64) {
//*prob_e += adjust_e;
//let zero = TpU64x8::ZERO;
//let max = TpU64x8::MAX;
//let adjust_e_ = TpI64::protect(adjust_e);
//unsafe {
//let shr: TpU64x8 = adjust_e_.tp_gt_eq(&64).select(zero, transmute(*prob >> adjust_e));
//let shl: TpU64x8 = adjust_e_.tp_lt_eq(&-64).select(max, transmute(*prob << -adjust_e));
//*prob = transmute(adjust_e_.tp_gt(&0).select(shr, shl));
//}

//}

pub fn renorm_scale_single<const F: usize>(prob: &mut i64x8, prob_e: &mut i64x8) {
    let leading_zeros: i64x8 = unsafe { std::mem::transmute(prob.leading_zeros()) };
    let adjust_e = i64x8::splat(64) - leading_zeros - i64x8::splat(F as i64);
    adjust_scale_single::<F>(adjust_e, prob, prob_e);
}

pub fn adjust_scale_single<const F: usize>(adjust_e: i64x8, prob: &mut i64x8, probs_e: &mut i64x8) {
    *probs_e += adjust_e;
    let shr = *prob >> adjust_e;
    let shl = *prob << -adjust_e;
    for (((&e, i), &l), &r) in adjust_e
        .as_array()
        .iter()
        .zip(prob.as_mut_array().iter_mut())
        .zip(shl.as_array().iter())
        .zip(shr.as_array().iter())
    {
        let e = TpI64::protect(e);
        unsafe {
            let r = e.tp_gt_eq(&64).select(TpFixed64::<F>::ZERO, transmute(r));
            let l = e.tp_lt_eq(&-64).select(TpFixed64::<F>::NAN, transmute(l));
            *i = transmute(e.tp_gt(&0).select(r, l));
        }
    }
}
