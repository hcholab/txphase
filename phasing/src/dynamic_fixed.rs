use crate::Real;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};
use tp_fixedpoint::timing_shield::{TpBool, TpI16, TpOrd};
use tp_fixedpoint::TpFixed64;

pub fn adjust_scale_single<const F: usize>(e: TpI16, prob: &mut TpFixed64<F>, probs_e: &mut TpI16) {
    let do_scale_up = e.tp_gt(&0);
    *probs_e += e;
    _adjust_scale(e, do_scale_up, prob);
}

// probs * 2 ^ probs_e -> (probs * 2 ^ -e) * 2 ^ (probs_e + e)
pub fn adjust_scale_row<const F: usize>(
    e: TpI16,
    mut probs: ArrayViewMut1<TpFixed64<F>>,
    probs_e: &mut TpI16,
) {
    let do_scale_up = e.tp_gt(&0);
    *probs_e += e;
    probs.iter_mut().for_each(|v| {
        _adjust_scale(e, do_scale_up, v);
    });
}

#[inline]
fn _adjust_scale<const F: usize>(e: TpI16, do_scale_up: TpBool, prob: &mut TpFixed64<F>) {
    let e = e.tp_gt_eq(&64).select(TpI16::protect(64), e);

    *prob = do_scale_up.select(
        e.tp_gt_eq(&64)
            .select(TpFixed64::<F>::ZERO, *prob >> e.expose() as u32),
        e.tp_lt_eq(&-64)
            .select(TpFixed64::<F>::NAN, *prob << ((-e).expose()) as u32),
    )
}

pub fn match_scale_single<const F: usize>(
    e_to_match: TpI16,
    probs: &mut TpFixed64<F>,
    probs_e: &mut TpI16,
) {
    let e = e_to_match - *probs_e;
    adjust_scale_single(e, probs, probs_e)
}

// probs * 2 ^ probs_e -> (probs * 2 ^ (probs_e-e_to_match)) * 2 ^ e_to_match
pub fn match_scale_row<const F: usize>(
    e_to_match: TpI16,
    probs: ArrayViewMut1<TpFixed64<F>>,
    probs_e: &mut TpI16,
) {
    let e = e_to_match - *probs_e;
    adjust_scale_row(e, probs, probs_e);
}

pub fn match_scale_arr1<const F: usize>(
    e_to_match: TpI16,
    mut probs: ArrayViewMut1<TpFixed64<F>>,
    mut probs_e: ArrayViewMut1<TpI16>,
) {
    Zip::from(&mut probs)
        .and(&mut probs_e)
        .for_each(|p, p_e| match_scale_single(e_to_match, p, p_e));
}

pub fn match_scale(
    e_to_match: TpI16,
    mut probs: ArrayViewMut2<Real>,
    mut probs_e: ArrayViewMut1<TpI16>,
) {
    Zip::from(probs.rows_mut())
        .and(&mut probs_e)
        .for_each(|probs_row, probs_e_i| match_scale_row(e_to_match, probs_row, probs_e_i));
}

pub fn renorm_scale_single<const F: usize>(prob: &mut TpFixed64<F>, prob_e: &mut TpI16) {
    let e = TpI16::protect(64) - prob.leading_zeros().as_i16() - TpI16::protect(F as i16);
    adjust_scale_single(e, prob, prob_e);
}

// renormalize scale so that largest number in probs is between 0.1 and 1.0
pub fn renorm_scale_row<const F: usize>(probs: ArrayViewMut1<TpFixed64<F>>, probs_e: &mut TpI16) {
    let min_leading_zeros = probs
        .iter()
        .cloned()
        .reduce(|accu, item| accu | item)
        .unwrap()
        .leading_zeros();

    let e = TpI16::protect(64) - min_leading_zeros.as_i16() - TpI16::protect(F as i16);
    adjust_scale_row(e, probs, probs_e);
}

pub fn renorm_scale_arr1<const F: usize>(
    mut probs: ArrayViewMut1<TpFixed64<F>>,
    mut probs_e: ArrayViewMut1<TpI16>,
) {
    Zip::from(&mut probs)
        .and(&mut probs_e)
        .for_each(|p, e| renorm_scale_single(p, e));
}

pub fn renorm_scale<const F: usize>(
    mut probs: ArrayViewMut2<TpFixed64<F>>,
    mut probs_e: ArrayViewMut1<TpI16>,
) {
    Zip::from(probs.rows_mut())
        .and(&mut probs_e)
        .for_each(|p, e| renorm_scale_row(p, e));
}

pub fn sum_scale_by_row(
    probs: ArrayView2<Real>,
    probs_e: ArrayView1<TpI16>,
) -> (Array1<Real>, Array1<TpI16>) {
    //debug_sum_by_row(probs, probs_e);
    let mut sum_by_row = Zip::from(probs.rows()).map_collect(|r| r.sum());
    let mut sum_by_row_e = probs_e.to_owned();
    renorm_scale_arr1(sum_by_row.view_mut(), sum_by_row_e.view_mut());
    (sum_by_row, sum_by_row_e)
}

//pub fn debug_sum_by_row(probs: ArrayView2<Real>, probs_e: ArrayView1<TpI16>) {
//Zip::from(probs.rows()).and(&probs_e).for_each(|r, &e| {
//if r.iter()
//.map(|v| v.into_inner().expose() as f64)
//.sum::<f64>()
//> i64::MAX as f64
//{
//println!("{:#?}", debug_expose_row(r, e));
//println!("{:#?}", e.expose());
//println!("{:#?}", debug_expose_s_arr1(r));
//}
//});
//}

pub fn sum_scale_by_column(
    probs: ArrayView2<Real>,
    probs_e: ArrayView1<TpI16>,
) -> (Array1<Real>, TpI16) {
    let mut sum_e = max_e(probs_e);

    let mut probs = probs.to_owned();
    let mut probs_e = probs_e.to_owned();
    Zip::from(probs.rows_mut())
        .and(&mut probs_e)
        .for_each(|p, e| {
            match_scale_row(sum_e, p, e);
        });

    let mut sum = Zip::from(probs.columns()).map_collect(|c| c.sum());

    renorm_scale_row(sum.view_mut(), &mut sum_e);

    (sum, sum_e)
}

pub fn sum_scale_arr1(
    mut probs: ArrayViewMut1<Real>,
    mut probs_e: ArrayViewMut1<TpI16>,
) -> (Real, TpI16) {
    let mut sum_e = max_e(probs_e.view());

    Zip::from(&mut probs).and(&mut probs_e).for_each(|p, e| {
        match_scale_single(sum_e, p, e);
    });

    let mut sum = probs.sum();
    renorm_scale_single(&mut sum, &mut sum_e);

    (sum, sum_e)
}

pub fn sum_scale(probs: ArrayView2<Real>, probs_e: ArrayView1<TpI16>) -> (Real, TpI16) {
    let (mut sum_by_row, mut sum_by_row_e) = sum_scale_by_row(probs.view(), probs_e.view());
    sum_scale_arr1(sum_by_row.view_mut(), sum_by_row_e.view_mut())
}

pub fn renorm_equalize_scale(
    mut probs: ArrayViewMut2<Real>,
    mut probs_e: ArrayViewMut2<TpI16>,
    mut equalized_probs_e: ArrayViewMut1<TpI16>,
) {
    Zip::from(probs.rows_mut())
        .and(probs_e.rows_mut())
        .and(&mut equalized_probs_e)
        .for_each(|p_row, p_row_e, tar_e| {
            *tar_e = renorm_equalize_scale_arr1(p_row, p_row_e);
        });
}

pub fn renorm_equalize_scale_all<const F: usize>(
    mut probs: ArrayViewMut2<TpFixed64<F>>,
    mut probs_e: ArrayViewMut2<TpI16>,
) -> TpI16 {
    let e_to_match =
        Zip::from(&probs)
            .and(&probs_e)
            .fold(TpI16::protect(i16::MIN), |accu, p, &e| {
                let new_e =
                    TpI16::protect(64) - p.leading_zeros().as_i16() - TpI16::protect(F as i16) + e;
                accu.tp_gt(&new_e).select(accu, new_e)
            });

    Zip::from(&mut probs)
        .and(&mut probs_e)
        .for_each(|p, e| match_scale_single(e_to_match, p, e));
    e_to_match
}

pub fn renorm_equalize_scale_arr1<const F: usize>(
    probs: ArrayViewMut1<TpFixed64<F>>,
    probs_e: ArrayViewMut1<TpI16>,
) -> TpI16 {
    let e_to_match =
        Zip::from(&probs)
            .and(&probs_e)
            .fold(TpI16::protect(i16::MIN), |accu, p, &e| {
                let new_e =
                    TpI16::protect(64) - p.leading_zeros().as_i16() - TpI16::protect(F as i16) + e;
                accu.tp_gt(&new_e).select(accu, new_e)
            });
    match_scale_arr1(e_to_match, probs, probs_e);
    e_to_match
}
pub fn renorm_e_pair(mut e_1: ArrayViewMut1<TpI16>, mut e_2: ArrayViewMut1<TpI16>) {
    let max_e_1 = min_e(e_1.view());
    let max_e_2 = min_e(e_2.view());
    let max_e = max_e_1.tp_gt(&max_e_2).select(max_e_1, max_e_2);
    e_1.map_mut(|v| *v -= max_e);
    e_2.map_mut(|v| *v -= max_e);
}

pub fn renorm_e(mut e: ArrayViewMut1<TpI16>) {
    let max_e = min_e(e.view());
    e.map_mut(|v| *v -= max_e);
}

pub fn min_e(e: ArrayView1<TpI16>) -> TpI16 {
    e.iter()
        .cloned()
        .reduce(|accu, item| (accu.tp_lt(&item)).select(accu, item))
        .unwrap()
}

pub fn max_e(e: ArrayView1<TpI16>) -> TpI16 {
    e.iter()
        .cloned()
        .reduce(|accu, item| (accu.tp_gt(&item)).select(accu, item))
        .unwrap()
}

pub fn debug_expose(s: Real, e: TpI16) -> f64 {
    s.expose_into_f32() as f64 * (e.expose() as f64).exp2()
}

pub fn debug_expose_row(s: ArrayView1<Real>, e: TpI16) -> Array1<f64> {
    let e = (e.expose() as f64).exp2();
    s.map(|v| v.expose_into_f32() as f64 * e)
}

pub fn debug_create_row(s: ArrayView1<f64>) -> (Array1<Real>, TpI16) {
    let s_ = Array1::<Real>::from_elem(s.raw_dim(), Real::ZERO);
    let e = TpI16::protect(0);
    (s_, e)
}

pub fn debug_create_array(s: ArrayView2<f64>) -> (Array2<Real>, Array1<TpI16>) {
    let s_ = Array2::<Real>::from_elem(s.raw_dim(), Real::ZERO);
    let e = Array1::<TpI16>::from_elem(s.nrows(), TpI16::protect(0));
    (s_, e)
}

pub fn debug_convert_row(s: ArrayView1<f64>) -> (Array1<Real>, TpI16) {
    let log2 = s
        .iter()
        .into_iter()
        .cloned()
        .reduce(|accu, item| accu.max(item))
        .unwrap()
        .log2()
        .ceil();
    let e = TpI16::protect(log2 as i16);
    (s.map(|v| Real::protect_f32((v / log2.exp2()) as f32)), e)
}

pub fn debug_convert_array(s: ArrayView2<f64>) -> (Array2<Real>, Array1<TpI16>) {
    let (mut s_, mut e) = debug_create_array(s);
    Zip::from(s.rows())
        .and(s_.rows_mut())
        .and(&mut e)
        .for_each(|sr, mut sr_, ei| {
            let (sr, ei_) = debug_convert_row(sr);
            *ei = ei_;
            sr_.assign(&sr);
        });
    (s_, e)
}

pub fn debug_expose_array(s: ArrayView2<Real>, e: ArrayView1<TpI16>) -> Array2<f64> {
    let mut s_out = Array2::<f64>::zeros(s.dim());
    Zip::from(s_out.rows_mut())
        .and(s.rows())
        .and(&e)
        .for_each(|mut o, s_row, &e_i| o.assign(&debug_expose_row(s_row, e_i)));
    s_out
}

pub fn debug_expose_array_ext(s: ArrayView2<Real>, e: ArrayView2<TpI16>) -> Array2<f64> {
    Zip::from(&s)
        .and(&e)
        .map_collect(|&_s, &_e| debug_expose(_s, _e))
}

pub fn debug_expose_s(s: ArrayView2<Real>) -> Array2<f64> {
    s.map(|v| v.expose_into_f32() as f64)
}
pub fn debug_expose_s_arr1(s: ArrayView1<Real>) -> Array1<f64> {
    s.map(|v| v.expose_into_f32() as f64)
}

pub fn debug_expose_e(e: ArrayView1<TpI16>) -> Array1<i16> {
    e.map(|v| v.expose())
}
