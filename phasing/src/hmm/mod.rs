pub mod params;
pub use params::*;

#[cfg(feature = "obliv")]
use crate::dynamic_fixed::*;
use crate::genotype_graph::{G, P};
use crate::{Bool, Genotype, Real};
#[cfg(feature = "obliv")]
use ndarray::{Array1, ArrayViewMut1};
#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::{TpEq, TpI16};

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

lazy_static::lazy_static! {
    pub static ref EMIS_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
    pub static ref TRAN_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
    pub static ref COLL_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
    pub static ref COMB_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
    pub static ref COMBD_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
}

use ndarray::{s, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, Zip};
pub fn combine_dips(
    tprobs: ArrayView2<Real>,
    #[cfg(feature = "obliv")] tprobs_e: ArrayView2<TpI16>,
    tprobs_dips: ArrayViewMut2<Real>,
    #[cfg(feature = "obliv")] cond: Bool,
) {
    let t = Instant::now();

    #[cfg(not(feature = "obliv"))]
    let mut tprobs_dips = tprobs_dips;

    #[cfg(not(feature = "obliv"))]
    let tprobs = &tprobs * (1. / tprobs.sum());

    #[cfg(feature = "obliv")]
    let mut _tprobs_dips = tprobs_dips;

    #[cfg(feature = "obliv")]
    let mut tprobs_dips = Array2::<Real>::zeros((P, P));

    #[cfg(feature = "obliv")]
    let mut tprobs_dips_e_ext = Array2::<TpI16>::from_elem((P, P), TpI16::protect(0));

    for i in 0..P {
        for j in 0..P {
            tprobs_dips[[i, j]] = tprobs[[i, j]] * tprobs[[P - 1 - i, P - 1 - j]];
            #[cfg(feature = "obliv")]
            {
                tprobs_dips_e_ext[[i, j]] = tprobs_e[[i, j]] + tprobs_e[[P - 1 - i, P - 1 - j]];
            }
        }
    }

    #[cfg(feature = "obliv")]
    {
        renorm_equalize_scale_all(tprobs_dips.view_mut(), tprobs_dips_e_ext.view_mut());
         //let mut tprobs_dips_e = Array1::<TpI16>::from_elem(P, TpI16::protect(0));
            //renorm_equalize_scale(
                //tprobs_dips.view_mut(),
                //tprobs_dips_e_ext.view_mut(),
                //tprobs_dips_e.view_mut(),
            //);

        //let (sum, sum_e) = sum_scale(tprobs_dips.view(), tprobs_dips_e.view());
        //tprobs_dips *= Real::protect_i64(1) / sum;
        //tprobs_dips_e.map_mut(|v| *v -= sum_e);
        //Zip::from(tprobs_dips.rows_mut())
            //.and(&mut tprobs_dips_e)
            //.for_each(|t, e| match_scale_row(TpI16::protect(0), t, e));

        ndarray::Zip::from(&mut _tprobs_dips)
            .and(&tprobs_dips)
            .for_each(|t, &s| *t = cond.select(s, *t));
    }

    #[cfg(not(feature = "obliv"))]
    {
        tprobs_dips /= tprobs_dips.sum();
    }

    let mut _t = COMBD_T.lock().unwrap();
    *_t += Instant::now() - t;
}

#[cfg(not(feature = "obliv"))]
pub struct Hmm {}

#[cfg(feature = "obliv")]
pub struct Hmm {
    pub bprobs_e: Array2<TpI16>,
    cur_fprobs_e: Array1<TpI16>,
    prev_fprobs_e: Array1<TpI16>,
    pub tprobs_e: Array3<TpI16>,
    pub cur_i: usize,
    is_backward: bool,
}

impl Hmm {
    #[cfg(feature = "obliv")]
    pub fn new() -> Self {
        Self {
            bprobs_e: Array2::from_elem((0, 0), TpI16::protect(0)),
            cur_fprobs_e: Array1::from_elem(P, TpI16::protect(0)),
            prev_fprobs_e: Array1::from_elem(P, TpI16::protect(0)),
            tprobs_e: Array3::from_elem((0, 0, 0), TpI16::protect(0)),
            cur_i: 0,
            is_backward: true,
        }
    }
    #[cfg(feature = "obliv")]
    pub fn get_cur_probs_e(&mut self) -> ArrayViewMut1<TpI16> {
        if self.is_backward {
            self.bprobs_e.slice_mut(s![self.cur_i, ..])
        } else {
            self.cur_fprobs_e.view_mut()
        }
    }

    #[cfg(feature = "obliv")]
    pub fn get_probs_e(&mut self) -> (ArrayViewMut1<TpI16>, ArrayViewMut1<TpI16>) {
        if self.is_backward {
            self.bprobs_e
                .multi_slice_mut((s![self.cur_i, ..], s![self.cur_i + 1, ..]))
        } else {
            (self.cur_fprobs_e.view_mut(), self.prev_fprobs_e.view_mut())
        }
    }

    #[cfg(not(feature = "obliv"))]
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward_backward(
        &mut self,
        ref_panel: ArrayView2<Genotype>,
        genograph: ArrayView1<G>,
        hmm_params: &HmmParams,
        rprobs: &RprobsSlice,
        ignored_sites: ArrayView1<Bool>,
        is_first_window: bool,
    ) -> Array3<Real> {
        let m = ref_panel.nrows();
        let k = ref_panel.ncols();
        let mut rprobs_iter = rprobs.get_forward();

        let bprobs = self.backward(ref_panel, genograph, hmm_params, rprobs, ignored_sites);

        #[cfg(feature = "obliv")]
        {
            self.is_backward = false;
        }

        let mut tprobs = Array3::<Real>::zeros((m, P, P));

        #[cfg(feature = "obliv")]
        {
            self.tprobs_e = Array3::<TpI16>::from_elem((m, P, P), TpI16::protect(0));
        }

        if is_first_window {
            self.first_combine(bprobs.slice(s![0, .., ..]), tprobs.slice_mut(s![0, .., ..]));
        }

        let mut prev_fprobs = Array2::<Real>::zeros((P, k));
        let mut cur_fprobs = Array2::<Real>::zeros((P, k));

        self.init(
            ref_panel.slice(s![0, ..]),
            genograph[0],
            hmm_params,
            prev_fprobs.view_mut(),
        );

        #[cfg(feature = "obliv")]
        self.prev_fprobs_e.assign(&self.cur_fprobs_e);

        #[cfg(feature = "obliv")]
        let mut tprobs_3x = Array3::<Real>::zeros((m.div_ceil(3), P, P));
        #[cfg(feature = "obliv")]
        let mut tprobs_3x_e = Array3::<TpI16>::from_elem((m.div_ceil(3), P, P), TpI16::protect(0));
        #[cfg(feature = "obliv")]
        let mut tprobs_3x_fwd = prev_fprobs.clone();
        #[cfg(feature = "obliv")]
        let mut tprobs_3x_fwd_e = self.prev_fprobs_e.clone();
        #[cfg(feature = "obliv")]
        let mut tprobs_3x_bwd = bprobs.slice(s![0, .., ..]).to_owned();
        #[cfg(feature = "obliv")]
        let mut tprobs_3x_bwd_e = self.bprobs_e.row(0).to_owned();

        #[cfg(feature = "obliv")]
        {
            let cond = genograph[0].is_segment_marker();
            Zip::from(&mut tprobs_3x_fwd)
                .and(&prev_fprobs)
                .for_each(|t, &f| *t = cond.select(f, *t));
            Zip::from(&mut tprobs_3x_fwd_e)
                .and(&self.prev_fprobs_e)
                .for_each(|t, &f| *t = cond.select(f, *t));
            Zip::from(&mut tprobs_3x_bwd)
                .and(&bprobs.slice(s![0, .., ..]))
                .for_each(|t, &b| *t = cond.select(b, *t));
            Zip::from(&mut tprobs_3x_bwd_e)
                .and(&self.bprobs_e.slice(s![0, ..]))
                .for_each(|t, &b| *t = cond.select(b, *t));
        }

        #[cfg(not(feature = "obliv"))]
        if genograph[0].is_segment_marker() {
            Self::combine(
                prev_fprobs.view(),
                bprobs.slice(s![0, .., ..]),
                tprobs.slice_mut(s![0, .., ..]),
            );
            self.collapse(cur_fprobs.view_mut());
        }

        for i in 1..m {
            let rprobs = rprobs_iter.next().unwrap();

            #[cfg(not(feature = "obliv"))]
            if ignored_sites[i] {
                continue;
            }

            #[cfg(feature = "obliv")]
            {
                self.cur_i = i;
            }

            self.transition(rprobs, prev_fprobs.view(), cur_fprobs.view_mut());

            #[cfg(feature = "obliv")]
            {
                let cond = genograph[i].is_segment_marker();

                let bprobs_e = self.bprobs_e.slice_mut(s![i, ..]);
                let fprobs_e = self.cur_fprobs_e.view_mut();

                Zip::from(&mut tprobs_3x_fwd)
                    .and(&cur_fprobs)
                    .for_each(|t, &f| *t = cond.select(f, *t));

                Zip::from(&mut tprobs_3x_fwd_e)
                    .and(&fprobs_e)
                    .for_each(|t, &f| *t = cond.select(f, *t));

                Zip::from(&mut tprobs_3x_bwd)
                    .and(&bprobs.slice(s![i, .., ..]))
                    .for_each(|t, &b| *t = cond.select(b, *t));

                Zip::from(&mut tprobs_3x_bwd_e)
                    .and(&bprobs_e)
                    .for_each(|t, &b| *t = cond.select(b, *t));

                if i % 3 == 2 || i == m - 1 {
                    Self::combine(
                        tprobs_3x_fwd.view(),
                        tprobs_3x_fwd_e.view(),
                        tprobs_3x_bwd.view(),
                        tprobs_3x_bwd_e.view(),
                        tprobs_3x.slice_mut(s![i / 3, .., ..]),
                        tprobs_3x_e.slice_mut(s![i / 3, .., ..]),
                    );
                };

                let tmp = cur_fprobs.clone();
                let tmp_e = self.cur_fprobs_e.clone();
                self.collapse(cur_fprobs.view_mut());

                Zip::from(&tmp)
                    .and(&mut cur_fprobs)
                    .for_each(|t, c| *c = cond.select(*c, *t));

                Zip::from(&tmp_e)
                    .and(&mut self.cur_fprobs_e)
                    .for_each(|t, c| *c = cond.select(*c, *t));
            }

            #[cfg(not(feature = "obliv"))]
            if genograph[i].is_segment_marker() {
                Self::combine(
                    cur_fprobs.view(),
                    bprobs.slice(s![i, .., ..]),
                    tprobs.slice_mut(s![i, .., ..]),
                );
                self.collapse(cur_fprobs.view_mut());
            }

            self.emission(
                ref_panel.slice(s![i, ..]),
                genograph[i],
                #[cfg(not(feature = "obliv"))]
                hmm_params,
                cur_fprobs.view_mut(),
            );

            #[cfg(feature = "obliv")]
            {
                let cond = !ignored_sites[i];
                Zip::from(&cur_fprobs)
                    .and(&mut prev_fprobs)
                    .for_each(|c, p| *p = cond.select(*c, *p));

                Zip::from(&self.cur_fprobs_e)
                    .and(&mut self.prev_fprobs_e)
                    .for_each(|c, p| *p = cond.select(*c, *p));
            }

            #[cfg(not(feature = "obliv"))]
            prev_fprobs.assign(&cur_fprobs);
        }

        #[cfg(feature = "obliv")]
        for i in 0..m {
            tprobs
                .slice_mut(s![i, .., ..])
                .assign(&tprobs_3x.slice(s![i / 3, .., ..]));
            self.tprobs_e
                .slice_mut(s![i, .., ..])
                .assign(&tprobs_3x_e.slice(s![i / 3, .., ..]));
        }

        tprobs
    }

    fn backward(
        &mut self,
        ref_panel: ArrayView2<Genotype>,
        genograph: ArrayView1<G>,
        hmm_params: &HmmParams,
        rprobs: &RprobsSlice,
        ignored_sites: ArrayView1<Bool>,
    ) -> Array3<Real> {
        let m = ref_panel.nrows();
        let n = ref_panel.ncols();
        let mut rprobs_iter = rprobs.get_backward();

        let mut bprobs = Array3::<Real>::zeros((m, P, n));

        #[cfg(feature = "obliv")]
        {
            self.bprobs_e = Array2::<TpI16>::from_elem((m, P), TpI16::protect(0));
            self.cur_i = m - 1;
        }

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

            #[cfg(not(feature = "obliv"))]
            if ignored_sites[i] && !genograph[i + 1].is_segment_marker() {
                cur_bprob.assign(&prev_bprob);
                continue;
            }

            #[cfg(feature = "obliv")]
            {
                self.cur_i = i;
            }

            self.transition(rprobs, prev_bprob.view(), cur_bprob.view_mut());

            #[cfg(feature = "obliv")]
            {
                let cond = genograph[i + 1].is_segment_marker();
                let tmp = cur_bprob.to_owned();
                let tmp_e = self.get_cur_probs_e().to_owned();
                self.collapse(cur_bprob.view_mut());
                Zip::from(&tmp)
                    .and(&mut cur_bprob)
                    .for_each(|t, c| *c = cond.select(*c, *t));

                Zip::from(&tmp_e)
                    .and(&mut self.get_cur_probs_e())
                    .for_each(|t, c| *c = cond.select(*c, *t));
            }

            #[cfg(not(feature = "obliv"))]
            if genograph[i + 1].is_segment_marker() {
                self.collapse(cur_bprob.view_mut());
            }

            self.emission(
                ref_panel.row(i),
                genograph[i],
                #[cfg(not(feature = "obliv"))]
                hmm_params,
                cur_bprob.view_mut(),
            );

            #[cfg(feature = "obliv")]
            {
                let cond = !ignored_sites[i] | genograph[i + 1].is_segment_marker();
                Zip::from(&mut cur_bprob)
                    .and(&prev_bprob)
                    .for_each(|c, p| *c = cond.select(*c, *p));

                let (mut cur_bprob_e, prev_bprob_e) = self.get_probs_e();

                Zip::from(&mut cur_bprob_e)
                    .and(&prev_bprob_e)
                    .for_each(|c, p| *c = cond.select(*c, *p));
            }
        }

        bprobs
    }

    fn init(
        &mut self,
        cond_haps: ArrayView1<Genotype>,
        graph_col: G,
        hmm_params: &HmmParams,
        mut probs: ArrayViewMut2<Real>,
    ) {
        Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
            Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
                #[cfg(feature = "obliv")]
                {
                    let g = graph_col.get_row(i);
                    *p = (z.tp_not_eq(&g)).select(hmm_params.eprob, Real::protect_i64(1));
                }

                #[cfg(not(feature = "obliv"))]
                if z != graph_col.get_row(i) {
                    *p = hmm_params.eprob;
                } else {
                    *p = 1.;
                }
            });
        });

        #[cfg(feature = "obliv")]
        renorm_scale(probs, self.get_cur_probs_e());
    }

    fn emission(
        &mut self,
        cond_haps: ArrayView1<Genotype>,
        graph_col: G,
        #[cfg(not(feature = "obliv"))] hmm_params: &HmmParams,
        mut probs: ArrayViewMut2<Real>,
    ) {
        let t = Instant::now();

        Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
            Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
                #[cfg(feature = "obliv")]
                {
                    let g = graph_col.get_row(i);
                    *p = (z.tp_not_eq(&g))
                        .select((*p >> 14) + (*p >> 15) + (*p >> 17) + (*p >> 20), *p);
                }

                #[cfg(not(feature = "obliv"))]
                if z != graph_col.get_row(i) {
                    *p *= hmm_params.eprob;
                }
            });
        });

        #[cfg(feature = "obliv")]
        renorm_scale(probs, self.get_cur_probs_e());

        let mut _t = EMIS_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn transition(
        &mut self,
        rprob: Real,
        prev_probs: ArrayView2<Real>,
        mut cur_probs: ArrayViewMut2<Real>,
    ) {
        let t = Instant::now();
        #[cfg(feature = "obliv")]
        let (mut cur_probs_e, prev_probs_e) = self.get_probs_e();

        let all_sum_h = Zip::from(prev_probs.rows()).map_collect(|r| r.sum());

        #[cfg(feature = "obliv")]
        let all_sum_h_e = prev_probs_e.to_owned();

        let _all_sum_h = &all_sum_h * rprob;

        Zip::from(cur_probs.rows_mut())
            .and(prev_probs.rows())
            .and(&_all_sum_h)
            .for_each(|mut cr, pr, &s| cr.assign(&(&pr + s)));

        #[cfg(feature = "obliv")]
        {
            cur_probs_e.assign(&all_sum_h_e);
            renorm_scale(cur_probs.view_mut(), cur_probs_e.view_mut());
        }

        #[cfg(not(feature = "obliv"))]
        {
            cur_probs /= all_sum_h.sum();
        }

        let mut _t = TRAN_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn collapse(&mut self, mut cur_probs: ArrayViewMut2<Real>) {
        let t = Instant::now();

        #[cfg(feature = "obliv")]
        let (sum_k, sum_k_e) = sum_scale_by_column(cur_probs.view(), self.get_cur_probs_e().view());

        #[cfg(not(feature = "obliv"))]
        let mut sum_k = Zip::from(cur_probs.columns()).map_collect(|c| c.sum());

        #[cfg(not(feature = "obliv"))]
        {
            sum_k /= sum_k.sum();
        }

        #[cfg(feature = "obliv")]
        self.get_cur_probs_e().fill(sum_k_e);

        Zip::from(cur_probs.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_k));
        let mut _t = COLL_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn first_combine(
        &mut self,
        first_bprobs: ArrayView2<Real>,
        mut first_tprobs: ArrayViewMut2<Real>,
    ) {
        let init_probs = Zip::from(first_bprobs.rows()).map_collect(|r| r.sum());

        #[cfg(feature = "obliv")]
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

    #[cfg(feature = "obliv")]
    fn combine(
        fprobs: ArrayView2<Real>,
        fprobs_e: ArrayView1<TpI16>,
        bprobs: ArrayView2<Real>,
        bprobs_e: ArrayView1<TpI16>,
        mut tprobs: ArrayViewMut2<Real>,
        mut tprobs_e: ArrayViewMut2<TpI16>,
    ) {
        let t = Instant::now();

        Real::matmul(fprobs, bprobs.t(), tprobs.view_mut());

        Zip::from(tprobs_e.rows_mut())
            .and(&fprobs_e)
            .for_each(|mut t_r, &f_e| {
                Zip::from(&mut t_r).and(&bprobs_e).for_each(|t, &b_e| {
                    *t = f_e + b_e;
                })
            });

        Zip::from(&mut tprobs)
            .and(&mut tprobs_e)
            .for_each(|t, e| renorm_scale_single(t, e));

        let mut _t = COMB_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    #[cfg(not(feature = "obliv"))]
    fn combine(
        fprobs: ArrayView2<Real>,
        bprobs: ArrayView2<Real>,
        mut tprobs: ArrayViewMut2<Real>,
    ) {
        let t = Instant::now();
        tprobs.assign(&fprobs.dot(&bprobs.t()));
        let mut _t = COMB_T.lock().unwrap();
        *_t += Instant::now() - t;
    }
}
