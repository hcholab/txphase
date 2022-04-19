mod params;
pub use params::*;

use crate::genotype_graph::{G, P};
use crate::tp_value_real;
use crate::{Genotype, Real, RealHmm};
#[cfg(feature = "leak-resist-new")]
use ndarray::ArrayViewMut1;
#[cfg(feature = "leak-resist-new")]
use timing_shield::{TpBool, TpI16, TpOrd};

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, Zip};

#[cfg(not(feature = "leak-resist-new"))]
pub struct Hmm {}

#[cfg(feature = "leak-resist-new")]
pub struct Hmm {
    pub bprobs_e: Array2<TpI16>,
    cur_fprobs_e: Array1<TpI16>,
    prev_fprobs_e: Array1<TpI16>,
    pub cur_i: usize,
    is_backward: bool,
}

impl Hmm {
    #[cfg(feature = "leak-resist-new")]
    pub fn new() -> Self {
        Self {
            bprobs_e: Array2::from_elem((0, 0), TpI16::protect(0)),
            cur_fprobs_e: Array1::from_elem(P, TpI16::protect(0)),
            prev_fprobs_e: Array1::from_elem(P, TpI16::protect(0)),
            cur_i: 0,
            is_backward: true,
        }
    }
    #[cfg(feature = "leak-resist-new")]
    pub fn get_cur_probs_e(&mut self) -> ArrayViewMut1<TpI16> {
        if self.is_backward {
            self.bprobs_e.slice_mut(s![self.cur_i, ..])
        } else {
            self.cur_fprobs_e.view_mut()
        }
    }

    #[cfg(feature = "leak-resist-new")]
    pub fn get_probs_e(&mut self) -> (ArrayViewMut1<TpI16>, ArrayViewMut1<TpI16>) {
        if self.is_backward {
            self.bprobs_e
                .multi_slice_mut((s![self.cur_i, ..], s![self.cur_i + 1, ..]))
        } else {
            (self.cur_fprobs_e.view_mut(), self.prev_fprobs_e.view_mut())
        }
    }

    #[cfg(not(feature = "leak-resist-new"))]
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward_backward(
        &mut self,
        ref_panel: ArrayView2<Genotype>,
        genograph: ArrayView1<G>,
        hmm_params: &HmmParams,
        rprobs: &RprobsSlice,
        ignored_sites: ArrayView1<bool>,
        is_first_window: bool,
    ) -> Array3<RealHmm> {
        let m = ref_panel.nrows();
        let k = ref_panel.ncols();
        let mut rprobs_iter = rprobs.get_forward();

        let bprobs = self.backward(ref_panel, genograph, hmm_params, rprobs, ignored_sites);

        #[cfg(feature = "leak-resist-new")]
        {
            self.is_backward = false;
        }

        let mut tprobs = Array3::<RealHmm>::zeros((m, P, P));

        if is_first_window {
            self.first_combine(bprobs.slice(s![0, .., ..]), tprobs.slice_mut(s![0, .., ..]));
        }

        let mut prev_fprobs = Array2::<RealHmm>::zeros((P, k));
        let mut cur_fprobs = Array2::<RealHmm>::zeros((P, k));

        self.init(
            ref_panel.slice(s![0, ..]),
            genograph[0],
            hmm_params,
            prev_fprobs.view_mut(),
        );

        for i in 1..m {
            let rprobs = rprobs_iter.next().unwrap();

            #[cfg(feature = "leak-resist-new")]
            {
                self.cur_i = i;
            }

            if !ignored_sites[i] {
                self.transition(rprobs, prev_fprobs.view(), cur_fprobs.view_mut());

                if genograph[i].is_segment_marker() {
                    self.combine(
                        cur_fprobs.view(),
                        bprobs.slice(s![i, .., ..]),
                        tprobs.slice_mut(s![i, .., ..]),
                    );
                    self.collapse(cur_fprobs.view_mut());
                }

                self.emission(
                    ref_panel.slice(s![i, ..]),
                    genograph[i],
                    hmm_params,
                    cur_fprobs.view_mut(),
                );

                prev_fprobs.assign(&cur_fprobs);
                #[cfg(feature = "leak-resist-new")]
                self.prev_fprobs_e.assign(&self.cur_fprobs_e);
            }
        }
        tprobs
    }

    pub fn combine_dips(&mut self, tprobs: ArrayView2<RealHmm>, tprobs_dips: ArrayViewMut2<Real>) {
        #[cfg(not(feature = "leak-resist-new"))]
        let mut tprobs_dips = tprobs_dips;

        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_e = self.bprobs_e.row_mut(self.cur_i);

        #[cfg(feature = "leak-resist-new")]
        let (sum, sum_e) = sum_scale(tprobs.view(), tprobs_e.view());

        #[cfg(not(feature = "leak-resist-new"))]
        let sum = tprobs.sum();

        let scale = tp_value_real!(1, i64) / sum;

        let tprobs = &tprobs * scale;

        #[cfg(feature = "leak-resist-new")]
        let tprobs = {
            let mut tprobs = tprobs;
            tprobs_e.iter_mut().for_each(|v| *v -= sum_e);
            renorm_scale(tprobs.view_mut(), tprobs_e.view_mut());
            tprobs
        };

        #[cfg(feature = "leak-resist-new")]
        let mut _tprobs_dips = tprobs_dips;

        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_dips = Array2::<RealHmm>::zeros((P, P));

        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_dips_e = Array1::<TpI16>::from_elem(P, TpI16::protect(0));

        for i in 0..P {
            for j in 0..P {
                tprobs_dips[[i, j]] = tprobs[[i, j]] * tprobs[[P - 1 - i, P - 1 - j]];
            }

            #[cfg(feature = "leak-resist-new")]
            {
                tprobs_dips_e[i] = tprobs_e[i] + tprobs_e[P - 1 - i];
                renorm_scale_row(tprobs_dips.row_mut(i), &mut tprobs_dips_e[i]);
            }
        }

        #[cfg(feature = "leak-resist-new")]
        let (sum, sum_e) = sum_scale(tprobs_dips.view(), tprobs_dips_e.view());

        #[cfg(not(feature = "leak-resist-new"))]
        let sum = tprobs_dips.sum();

        let scale = tp_value_real!(1, i64) / sum;

        tprobs_dips *= scale;

        #[cfg(feature = "leak-resist-new")]
        {
            tprobs_dips_e.iter_mut().for_each(|v| *v -= sum_e);
            renorm_scale(tprobs_dips.view_mut(), tprobs_dips_e.view_mut());
            _tprobs_dips.assign(&debug_expose_array(
                tprobs_dips.view(),
                tprobs_dips_e.view(),
            ));
        }
    }

    fn backward(
        &mut self,
        ref_panel: ArrayView2<Genotype>,
        genograph: ArrayView1<G>,
        hmm_params: &HmmParams,
        rprobs: &RprobsSlice,
        ignored_sites: ArrayView1<bool>,
    ) -> Array3<RealHmm> {
        let m = ref_panel.nrows();
        let n = ref_panel.ncols();
        let mut rprobs_iter = rprobs.get_backward();

        let mut bprobs = Array3::<RealHmm>::zeros((m, P, n));

        #[cfg(feature = "leak-resist-new")]
        {
            self.bprobs_e = Array2::<TpI16>::from_elem((m, P), TpI16::protect(0));
            self.cur_i = m - 1;
        }

        //let mut last_cm = tp_value_real!(variants[m - 1].cm, f32);

        self.init(
            ref_panel.row(m - 1),
            genograph[m - 1],
            hmm_params,
            bprobs.slice_mut(s![m - 1, .., ..]),
        );

        for i in (0..m - 1).rev() {
            let rprobs = rprobs_iter.next().unwrap();
            let (mut cur_bprob, prev_bprob) =
                bprobs.multi_slice_mut((s![i, .., ..], s![i + 1, .., ..]));

            #[cfg(feature = "leak-resist-new")]
            {
                self.cur_i = i;
            }

            if !ignored_sites[i] || genograph[i + 1].is_segment_marker() {
                self.transition(rprobs, prev_bprob.view(), cur_bprob.view_mut());

                if genograph[i + 1].is_segment_marker() {
                    self.collapse(cur_bprob.view_mut());
                }

                self.emission(
                    ref_panel.row(i),
                    genograph[i],
                    hmm_params,
                    cur_bprob.view_mut(),
                );
            } else {
                cur_bprob.assign(&prev_bprob);

                #[cfg(feature = "leak-resist-new")]
                {
                    let (mut cur_bprobs_e, prev_bprobs_e) = self.get_probs_e();
                    cur_bprobs_e.assign(&prev_bprobs_e);
                }
            }
        }
        bprobs
    }

    fn init(
        &mut self,
        cond_haps: ArrayView1<Genotype>,
        graph_col: G,
        hmm_params: &HmmParams,
        mut probs: ArrayViewMut2<RealHmm>,
    ) {
        Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
            Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
                if z != graph_col.get_row(i) {
                    *p = hmm_params.eprob;
                } else {
                    *p = tp_value_real!(1, i64);
                }
            });
        });

        #[cfg(feature = "leak-resist-new")]
        renorm_scale(probs, self.get_cur_probs_e());
    }

    fn emission(
        &mut self,
        cond_haps: ArrayView1<Genotype>,
        graph_col: G,
        hmm_params: &HmmParams,
        mut probs: ArrayViewMut2<RealHmm>,
    ) {
        Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
            Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
                if z != graph_col.get_row(i) {
                    *p *= hmm_params.eprob;
                }
            });
        });

        #[cfg(feature = "leak-resist-new")]
        renorm_scale(probs, self.get_cur_probs_e());
    }

    fn transition(
        &mut self,
        rprob: (RealHmm, RealHmm),
        prev_probs: ArrayView2<RealHmm>,
        mut cur_probs: ArrayViewMut2<RealHmm>,
    ) {
        #[cfg(feature = "leak-resist-new")]
        let (mut cur_probs_e, prev_probs_e) = self.get_probs_e();

        #[cfg(feature = "leak-resist-new")]
        let (mut all_sum_h, mut all_sum_h_e) =
            sum_scale_by_row(prev_probs.view(), prev_probs_e.view());

        #[cfg(not(feature = "leak-resist-new"))]
        let all_sum_h = Array1::from_shape_fn(prev_probs.nrows(), |i| prev_probs.row(i).sum());

        #[cfg(feature = "leak-resist-new")]
        let prev_probs = {
            let mut prev_probs = prev_probs.to_owned();
            let mut prev_probs_e = prev_probs_e.to_owned();
            Zip::from(prev_probs.rows_mut())
                .and(&mut prev_probs_e)
                .and(&all_sum_h_e)
                .for_each(|p, e, &e_to_match| {
                    match_scale_row(e_to_match, p, e);
                });
            prev_probs
        };

        Zip::from(cur_probs.rows_mut())
            .and(prev_probs.rows())
            .and(&all_sum_h)
            .for_each(|mut cur_p_row, prev_p_row, &sum_h| {
                Zip::from(&mut cur_p_row)
                    .and(&prev_p_row)
                    .for_each(|cp, &pp| {
                        *cp = pp * rprob.1 + sum_h * rprob.0;
                    });
            });

        #[cfg(feature = "leak-resist-new")]
        {
            cur_probs_e.assign(&all_sum_h_e);
            renorm_scale(cur_probs.view_mut(), cur_probs_e.view_mut());
        }

        #[cfg(feature = "leak-resist-new")]
        let (sum, sum_e) = sum_scale_arr1(all_sum_h.view_mut(), all_sum_h_e.view_mut());

        #[cfg(not(feature = "leak-resist-new"))]
        let sum: RealHmm = all_sum_h.sum();

        let scale = tp_value_real!(1, i64) / sum;

        cur_probs *= scale;

        #[cfg(feature = "leak-resist-new")]
        {
            cur_probs_e.iter_mut().for_each(|v| *v -= sum_e);
            renorm_scale(cur_probs.view_mut(), cur_probs_e.view_mut());
        }
    }

    fn collapse(&mut self, mut cur_probs: ArrayViewMut2<RealHmm>) {
        #[cfg(feature = "leak-resist-new")]
        let (mut sum_k, mut sum_k_e) =
            sum_scale_by_column(cur_probs.view(), self.get_cur_probs_e().view());

        #[cfg(not(feature = "leak-resist-new"))]
        let mut sum_k =
            Array1::<RealHmm>::from_shape_fn(cur_probs.ncols(), |i| cur_probs.column(i).sum());

        let sum = sum_k.sum();

        #[cfg(feature = "leak-resist-new")]
        let mut sum_e = sum_k_e;

        #[cfg(feature = "leak-resist-new")]
        let sum = {
            let mut sum = sum;
            renorm_scale_single(&mut sum, &mut sum_e);
            sum
        };

        let scale = tp_value_real!(1, i64) / sum;

        sum_k *= scale;

        #[cfg(feature = "leak-resist-new")]
        {
            sum_k_e -= sum_e;
            renorm_scale_row(sum_k.view_mut(), &mut sum_k_e);
            self.get_cur_probs_e().fill(sum_k_e);
        }

        Zip::from(cur_probs.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_k));
    }

    fn first_combine(
        &mut self,
        first_bprobs: ArrayView2<RealHmm>,
        mut first_tprobs: ArrayViewMut2<RealHmm>,
    ) {
        let init_probs = Array1::<RealHmm>::from_shape_fn(P, |i| first_bprobs.row(i).sum());

        #[cfg(feature = "leak-resist-new")]
        let init_probs = {
            let mut init_probs = init_probs;
            let mut first_bprobs_e = self.bprobs_e.row_mut(0);
            renorm_scale_arr1(init_probs.view_mut(), first_bprobs_e.view_mut());

            let e_to_match = max_e(first_bprobs_e.view());

            match_scale_arr1(e_to_match, init_probs.view_mut(), first_bprobs_e.view_mut());
            init_probs
        };

        Zip::from(first_tprobs.rows_mut()).for_each(|mut r| r.assign(&init_probs));
    }

    fn combine(
        &mut self,
        fprobs: ArrayView2<RealHmm>,
        bprobs: ArrayView2<RealHmm>,
        mut tprobs: ArrayViewMut2<RealHmm>,
    ) {
        #[cfg(feature = "leak-resist-new")]
        {
            let bprobs_e = self.bprobs_e.slice_mut(s![self.cur_i, ..]);
            let fprobs_e = self.cur_fprobs_e.view_mut();
            let mut tprobs_e = Array2::from_elem((P, P), TpI16::protect(0));
            Zip::from(tprobs.rows_mut())
                .and(tprobs_e.rows_mut())
                .and(fprobs.rows())
                .and(&fprobs_e)
                .for_each(|mut t_row, mut t_row_e, f_row, &f_e| {
                    Zip::from(&mut t_row)
                        .and(&mut t_row_e)
                        .and(bprobs.rows())
                        .and(&bprobs_e)
                        .for_each(|t, t_e, b_row, &b_e| {
                            let mut mult = &f_row * &b_row;
                            let mut e = f_e + b_e;
                            renorm_scale_row(mult.view_mut(), &mut e);
                            let mut dot_product = mult.sum();
                            renorm_scale_single(&mut dot_product, &mut e);
                            *t = dot_product;
                            *t_e = e;
                        });
                });
            equalize_scale(tprobs.view_mut(), tprobs_e.view_mut(), bprobs_e);
        }

        #[cfg(not(feature = "leak-resist-new"))]
        Zip::from(tprobs.rows_mut())
            .and(fprobs.rows())
            .for_each(|mut t_row, f_row| {
                Zip::from(&mut t_row)
                    .and(bprobs.rows())
                    .for_each(|t, b_row| *t = f_row.dot(&b_row));
            });
    }
}

#[cfg(feature = "leak-resist-new")]
pub use inner::*;

#[cfg(feature = "leak-resist-new")]
mod inner {
    use super::*;
    use tp_fixedpoint::TpFixed64;

    pub fn adjust_scale_single<const F: usize>(
        e: TpI16,
        prob: &mut TpFixed64<F>,
        probs_e: &mut TpI16,
    ) {
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
        *prob = do_scale_up.select(*prob >> e.expose() as u32, *prob << ((-e).expose()) as u32)
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

    pub fn match_scale_arr1(
        e_to_match: TpI16,
        mut probs: ArrayViewMut1<RealHmm>,
        mut probs_e: ArrayViewMut1<TpI16>,
    ) {
        Zip::from(&mut probs)
            .and(&mut probs_e)
            .for_each(|p, p_e| match_scale_single(e_to_match, p, p_e));
    }

    pub fn match_scale(
        e_to_match: TpI16,
        mut probs: ArrayViewMut2<RealHmm>,
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
    pub fn renorm_scale_row<const F: usize>(
        probs: ArrayViewMut1<TpFixed64<F>>,
        probs_e: &mut TpI16,
    ) {
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
        probs: ArrayView2<RealHmm>,
        probs_e: ArrayView1<TpI16>,
    ) -> (Array1<RealHmm>, Array1<TpI16>) {
        let mut sum_by_row = Array1::from_shape_fn(probs.nrows(), |i| probs.row(i).sum());
        let mut sum_by_row_e = probs_e.to_owned();
        renorm_scale_arr1(sum_by_row.view_mut(), sum_by_row_e.view_mut());
        (sum_by_row, sum_by_row_e)
    }

    pub fn sum_scale_by_column(
        probs: ArrayView2<RealHmm>,
        probs_e: ArrayView1<TpI16>,
    ) -> (Array1<RealHmm>, TpI16) {
        let mut sum_e = max_e(probs_e);

        let mut probs = probs.to_owned();
        let mut probs_e = probs_e.to_owned();
        Zip::from(probs.rows_mut())
            .and(&mut probs_e)
            .for_each(|p, e| {
                match_scale_row(sum_e, p, e);
            });

        let mut sum = Array1::<RealHmm>::from_shape_fn(probs.ncols(), |i| probs.column(i).sum());

        renorm_scale_row(sum.view_mut(), &mut sum_e);

        (sum, sum_e)
    }

    pub fn sum_scale_arr1(
        mut probs: ArrayViewMut1<RealHmm>,
        mut probs_e: ArrayViewMut1<TpI16>,
    ) -> (RealHmm, TpI16) {
        let mut sum_e = max_e(probs_e.view());

        Zip::from(&mut probs).and(&mut probs_e).for_each(|p, e| {
            match_scale_single(sum_e, p, e);
        });

        let mut sum = probs.sum();
        renorm_scale_single(&mut sum, &mut sum_e);

        (sum, sum_e)
    }

    pub fn sum_scale(probs: ArrayView2<RealHmm>, probs_e: ArrayView1<TpI16>) -> (RealHmm, TpI16) {
        let (mut sum_by_row, mut sum_by_row_e) = sum_scale_by_row(probs.view(), probs_e.view());
        sum_scale_arr1(sum_by_row.view_mut(), sum_by_row_e.view_mut())
    }

    pub fn equalize_scale(
        mut probs: ArrayViewMut2<RealHmm>,
        mut probs_e: ArrayViewMut2<TpI16>,
        mut equalized_probs_e: ArrayViewMut1<TpI16>,
    ) {
        Zip::from(probs.rows_mut())
            .and(probs_e.rows_mut())
            .and(&mut equalized_probs_e)
            .for_each(|mut p_row, mut p_row_e, tar_e| {
                *tar_e = max_e(p_row_e.view());
                match_scale_arr1(*tar_e, p_row.view_mut(), p_row_e.view_mut());
                renorm_scale_row(p_row, tar_e);
            });
    }

    pub fn max_e(e: ArrayView1<TpI16>) -> TpI16 {
        e.iter()
            .cloned()
            .reduce(|accu, item| (accu.tp_gt(&item)).select(accu, item))
            .unwrap()
    }

    pub fn debug_expose(s: RealHmm, e: TpI16) -> f64 {
        s.expose_into_f32() as f64 * (e.expose() as f64).exp2()
    }

    pub fn debug_expose_row(s: ArrayView1<RealHmm>, e: TpI16) -> Array1<f64> {
        let e = (e.expose() as f64).exp2();
        let n = s.len();
        Array1::from_shape_fn(n, |i| s[i].expose_into_f32() as f64 * e)
    }

    pub fn debug_expose_array(s: ArrayView2<RealHmm>, e: ArrayView1<TpI16>) -> Array2<f64> {
        let mut s_out = Array2::<f64>::zeros(s.dim());
        Zip::from(s_out.rows_mut())
            .and(s.rows())
            .and(&e)
            .for_each(|mut o, s_row, &e_i| o.assign(&debug_expose_row(s_row, e_i)));
        s_out
    }

    pub fn debug_expose_s(s: ArrayView2<RealHmm>) -> Array2<f64> {
        Array2::<f64>::from_shape_fn(s.dim(), |(i, j)| s[[i, j]].expose_into_f32() as f64)
    }

    pub fn debug_expose_e(e: ArrayView1<TpI16>) -> Array1<i16> {
        Array1::<i16>::from_shape_fn(e.dim(), |i| e[i].expose())
    }
}
