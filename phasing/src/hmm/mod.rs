mod params;
pub use params::*;

use crate::genotype_graph::{G, P};
use crate::tp_value_real;
use crate::{BoolMcc, Genotype, Real};
use ndarray::Array1;
#[cfg(feature = "leak-resist-new")]
use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpI16, TpI8, TpOrd};

use rand::Rng;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const RENORM_THESHOLD: f64 = 1e-20;

lazy_static::lazy_static! {
    pub static ref EMIS_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
    pub static ref TRAN_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
    pub static ref COLL_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
    pub static ref COMB_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
    pub static ref COMBD_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
}

use ndarray::{
    s, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, ArrayViewMut3, Zip,
};

#[cfg(not(feature = "leak-resist-new"))]
pub struct HmmNew {}

#[cfg(not(feature = "leak-resist-new"))]
impl HmmNew {
    pub fn fwbw_sample<R: Rng>(
        ref_panel: ArrayView2<Genotype>,
        genograph: ArrayView1<G>,
        eprob: Real,
        rprobs: &RprobsSlice,
        ignored_sites: ArrayView1<BoolMcc>,
        is_first_window: bool,
        sampling_args: Option<(Option<u8>, R, ArrayViewMut1<u8>)>,
        mut accu_tprobs_dip: Option<(Option<ArrayViewMut1<Real>>, ArrayViewMut3<Real>)>,
        is_first_prune_main: bool,
    ) -> Option<u8> {
        let m = ref_panel.nrows();
        let k = ref_panel.ncols();
        let mut rprobs_iter = rprobs.get_forward();

        let (prev_ind, mut rng, mut sampled_ind) =
            if let Some((prev_ind, rng, sampled_ind)) = sampling_args {
                (Some(prev_ind), Some(rng), Some(sampled_ind))
            } else {
                (None, None, None)
            };

        let (first_accu_tprobs_dip, mut accu_tprobs_dip) =
            if let Some((first_accu_tprobs_dip, accu_tprobs_dip)) = accu_tprobs_dip {
                (Some(first_accu_tprobs_dip), Some(accu_tprobs_dip))
            } else {
                (None, None)
            };
        let bprobs = Self::backward(ref_panel, genograph, eprob, rprobs, ignored_sites);

        let first_ind = if is_first_window {
            let (first_ind, first_tprobs_dip) = Self::first_combine_and_sample(
                bprobs.slice(s![0, .., ..]),
                rng.as_mut(),
                accu_tprobs_dip.is_some(),
            );

            if let Some(mut first_tprobs_dip) = first_tprobs_dip {
                let mut first_accu_tprobs_dip = first_accu_tprobs_dip.unwrap().unwrap();
                if is_first_prune_main {
                    first_accu_tprobs_dip.assign(&first_tprobs_dip);
                } else {
                    first_accu_tprobs_dip += &first_tprobs_dip;
                }
            }
            first_ind
        } else {
            None
        };

        let mut cur_ind = if is_first_window {
            first_ind
        } else {
            if let Some(prev_ind) = prev_ind {
                prev_ind
            } else {
                None
            }
        };

        let mut prev_fprobs = Array2::<Real>::zeros((P, k));
        let mut cur_fprobs = Array2::<Real>::zeros((P, k));

        Self::init(
            ref_panel.slice(s![0, ..]),
            genograph[0],
            eprob,
            prev_fprobs.view_mut(),
        );

        //#[cfg(feature = "leak-resist-new")]
        //self.prev_fprobs_e.assign(&self.cur_fprobs_e);

        //#[cfg(feature = "leak-resist-new")]
        //let mut tprobs_3x = Array3::<Real>::zeros(((m + 2) / 3, P, P));
        //#[cfg(feature = "leak-resist-new")]
        //let mut tprobs_3x_e = Array3::<TpI16>::from_elem(((m + 2) / 3, P, P), TpI16::protect(0));
        //#[cfg(feature = "leak-resist-new")]
        //let mut tprobs_3x_fwd = prev_fprobs.clone();
        //#[cfg(feature = "leak-resist-new")]
        //let mut tprobs_3x_fwd_e = self.prev_fprobs_e.clone();
        //#[cfg(feature = "leak-resist-new")]
        //let mut tprobs_3x_bwd = bprobs.slice(s![0, .., ..]).to_owned();
        //#[cfg(feature = "leak-resist-new")]
        //let mut tprobs_3x_bwd_e = self.bprobs_e.row(0).to_owned();

        let mut segment_i = 0;
        for i in 1..m {
            let rprobs = rprobs_iter.next().unwrap();

            #[cfg(not(feature = "leak-resist-new"))]
            if ignored_sites[i] {
                continue;
            }

            Self::transition(
                rprobs.0 / rprobs.1,
                prev_fprobs.view(),
                cur_fprobs.view_mut(),
            );

            //#[cfg(feature = "leak-resist-new")]
            //{
            //let cond = TpBool::protect(genograph[i].is_segment_marker());

            //let bprobs_e = self.bprobs_e.slice_mut(s![i, ..]);
            //let fprobs_e = self.cur_fprobs_e.view_mut();

            //Zip::from(&mut tprobs_3x_fwd)
            //.and(&cur_fprobs)
            //.for_each(|t, &f| *t = cond.select(f, *t));

            //Zip::from(&mut tprobs_3x_fwd_e)
            //.and(&fprobs_e)
            //.for_each(|t, &f| *t = cond.select(f, *t));

            //Zip::from(&mut tprobs_3x_bwd)
            //.and(&bprobs.slice(s![i, .., ..]))
            //.for_each(|t, &b| *t = cond.select(b, *t));

            //Zip::from(&mut tprobs_3x_bwd_e)
            //.and(&bprobs_e)
            //.for_each(|t, &b| *t = cond.select(b, *t));

            //if i % 3 == 2 {
            //Self::combine(
            //tprobs_3x_fwd.view(),
            //tprobs_3x_fwd_e.view(),
            //tprobs_3x_bwd.view(),
            //tprobs_3x_bwd_e.view(),
            //tprobs_3x.slice_mut(s![i / 3, .., ..]),
            //tprobs_3x_e.slice_mut(s![i / 3, .., ..]),
            //);
            //};

            //let tmp = cur_fprobs.clone();
            //let tmp_e = self.cur_fprobs_e.clone();
            //self.collapse(cur_fprobs.view_mut());

            //Zip::from(&tmp)
            //.and(&mut cur_fprobs)
            //.for_each(|t, c| *c = cond.select(*c, *t));

            //Zip::from(&tmp_e)
            //.and(&mut self.cur_fprobs_e)
            //.for_each(|t, c| *c = cond.select(*c, *t));
            //}

            #[cfg(not(feature = "leak-resist-new"))]
            if genograph[i].is_segment_marker() {
                let (next_ind, tprobs_dip_pos) = Self::combine_and_sample(
                    cur_fprobs.view(),
                    bprobs.slice(s![i, .., ..]),
                    rng.as_mut().map(|rng| (cur_ind.unwrap(), rng)),
                    accu_tprobs_dip.is_some(),
                );
                Self::save_sample_tprobs_dips(
                    segment_i,
                    next_ind,
                    sampled_ind.as_mut(),
                    tprobs_dip_pos,
                    accu_tprobs_dip.as_mut(),
                    is_first_prune_main,
                );

                Self::collapse(cur_fprobs.view_mut());

                cur_ind = next_ind;
                segment_i += 1;
            }

            Self::emission(
                ref_panel.slice(s![i, ..]),
                genograph[i],
                eprob,
                cur_fprobs.view_mut(),
            );

            //#[cfg(feature = "leak-resist-new")]
            //{
            //let cond = !ignored_sites[i];
            //Zip::from(&cur_fprobs)
            //.and(&mut prev_fprobs)
            //.for_each(|c, p| *p = cond.select(*c, *p));

            //Zip::from(&self.cur_fprobs_e)
            //.and(&mut self.prev_fprobs_e)
            //.for_each(|c, p| *p = cond.select(*c, *p));
            //}

            #[cfg(not(feature = "leak-resist-new"))]
            prev_fprobs.assign(&cur_fprobs);
            //}

            //#[cfg(feature = "leak-resist-new")]
            //for i in 0..m {
            //tprobs
            //.slice_mut(s![i, .., ..])
            //.assign(&tprobs_3x.slice(s![i / 3, .., ..]));
            //self.tprobs_e
            //.slice_mut(s![i, .., ..])
            //.assign(&tprobs_3x_e.slice(s![i / 3, .., ..]));
        }
        first_ind
    }

    fn save_sample_tprobs_dips(
        i: usize,
        ind: Option<u8>,
        save_ind: Option<&mut ArrayViewMut1<u8>>,
        tprobs_dip_pos: Option<Array2<Real>>,
        accu_tprobs: Option<&mut ArrayViewMut3<Real>>,
        is_first_prune_main: bool,
    ) {
        if let Some(ind) = ind {
            save_ind.unwrap()[i] = ind;
        }
        if let Some(tprobs_dip_pos) = tprobs_dip_pos {
            let mut t = accu_tprobs.unwrap().slice_mut(s![i, .., ..]);
            if is_first_prune_main {
                t.assign(&tprobs_dip_pos);
            } else {
                t += &tprobs_dip_pos;
            }
        }
    }

    fn backward(
        ref_panel: ArrayView2<Genotype>,
        genograph: ArrayView1<G>,
        eprob: Real,
        rprobs: &RprobsSlice,
        ignored_sites: ArrayView1<BoolMcc>,
    ) -> Array3<Real> {
        let m = ref_panel.nrows();
        let n = ref_panel.ncols();
        let mut rprobs_iter = rprobs.get_backward();

        let mut bprobs = Array3::<Real>::zeros((m, P, n));

        #[cfg(feature = "leak-resist-new")]
        let mut bprobs_e = Array2::<TpI16>::from_elem((m, P), TpI16::protect(0));

        Self::init(
            ref_panel.row(m - 1),
            genograph[m - 1],
            eprob,
            bprobs.slice_mut(s![m - 1, .., ..]),
        );

        for i in (0..m - 1).rev() {
            let rprobs = rprobs_iter.next().unwrap();
            let (mut cur_bprob, prev_bprob) =
                bprobs.multi_slice_mut((s![i, .., ..], s![i + 1, .., ..]));

            #[cfg(not(feature = "leak-resist-new"))]
            if ignored_sites[i] && !genograph[i + 1].is_segment_marker() {
                cur_bprob.assign(&prev_bprob);
                continue;
            }

            Self::transition(rprobs.0 / rprobs.1, prev_bprob.view(), cur_bprob.view_mut());

            #[cfg(feature = "leak-resist-new")]
            {
                let cond = TpBool::protect(genograph[i + 1].is_segment_marker());
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

            #[cfg(not(feature = "leak-resist-new"))]
            if genograph[i + 1].is_segment_marker() {
                Self::collapse(cur_bprob.view_mut());
            }

            Self::emission(ref_panel.row(i), genograph[i], eprob, cur_bprob.view_mut());

            #[cfg(feature = "leak-resist-new")]
            {
                let cond =
                    !ignored_sites[i] | TpBool::protect(genograph[i + 1].is_segment_marker());
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
        cond_haps: ArrayView1<Genotype>,
        graph: G,
        eprob: Real,
        mut probs: ArrayViewMut2<Real>,
        #[cfg(feature = "leak-resist-new")] probs_e: ArrayViewMut1<TpI16>,
    ) {
        let t = Instant::now();

        Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
            Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
                #[cfg(feature = "leak-resist-new")]
                {
                    let z = TpI8::protect(z);
                    let g = TpI8::protect(graph.get_row(i));
                    *p = (z.tp_not_eq(&g)).select(hmm_params.eprob, Real::protect_i64(1));
                }

                #[cfg(not(feature = "leak-resist-new"))]
                if z != graph.get_row(i) {
                    *p = eprob;
                } else {
                    *p = tp_value_real!(1, i64);
                }
            });
        });

        #[cfg(feature = "leak-resist-new")]
        renorm_scale(probs, probs_e);

        let mut _t = EMIS_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn emission(
        cond_haps: ArrayView1<Genotype>,
        graph_col: G,
        eprob: Real,
        mut probs: ArrayViewMut2<Real>,
        #[cfg(feature = "leak-resist-new")] probs_e: ArrayViewMut1<TpI16>,
    ) {
        let t = Instant::now();

        Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
            Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
                #[cfg(feature = "leak-resist-new")]
                {
                    let z = TpI8::protect(z);
                    let g = TpI8::protect(graph_col.get_row(i));
                    //p = (z.tp_not_eq(&g)).select(*p * hmm_params.eprob, *p);
                    *p = (z.tp_not_eq(&g))
                        .select((*p >> 14) + (*p >> 15) + (*p >> 17) + (*p >> 20), *p);
                }

                #[cfg(not(feature = "leak-resist-new"))]
                if z != graph_col.get_row(i) {
                    *p *= eprob;
                }
            });
        });

        #[cfg(feature = "leak-resist-new")]
        renorm_scale(probs, probs_e);

        let mut _t = EMIS_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn transition(
        rprob: Real,
        prev_probs: ArrayView2<Real>,
        #[cfg(feature = "leak-resist-new")] prev_probs_e: ArrayView1<TpI16>,
        mut cur_probs: ArrayViewMut2<Real>,
        #[cfg(feature = "leak-resist-new")] cur_probs_e: ArrayViewMut1<TpI16>,
    ) {
        let t = Instant::now();

        Zip::from(cur_probs.rows_mut())
            .and(prev_probs.rows())
            .for_each(|mut cr, pr| cr.assign(&(&pr + pr.sum() * rprob)));

        #[cfg(feature = "leak-resist-new")]
        {
            cur_probs_e.assign(&prev_probs_e);
            renorm_scale(cur_probs.view_mut(), cur_probs_e.view_mut());
        }

        #[cfg(not(feature = "leak-resist-new"))]
        {
            let sum = cur_probs.sum();
            if sum < RENORM_THESHOLD {
                cur_probs *= 1. / cur_probs.sum();
            }
        }

        let mut _t = TRAN_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn collapse(
        mut cur_probs: ArrayViewMut2<Real>,
        #[cfg(feature = "leak-resist-new")] cur_probs_e: ArrayViewMut1<TpI16>,
    ) {
        let t = Instant::now();

        #[cfg(feature = "leak-resist-new")]
        let (sum_k, sum_k_e) = sum_scale_by_column(cur_probs.view(), cur_probs_e.view());

        #[cfg(not(feature = "leak-resist-new"))]
        let sum_k = {
            let sum_k = Zip::from(cur_probs.columns()).map_collect(|c| c.sum());
            let sum = sum_k.sum();
            sum_k * 1. / sum
        };

        Zip::from(cur_probs.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_k));

        #[cfg(feature = "leak-resist-new")]
        cur_probs_e.fill(sum_k_e);

        let mut _t = COLL_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn first_combine_and_sample<R: Rng>(
        first_bprobs: ArrayView2<Real>,
        rng: Option<R>,
        save_tprobs_dip: bool,
    ) -> (Option<u8>, Option<Array1<Real>>) {
        let t = Instant::now();
        let mut init_tprobs = Zip::from(first_bprobs.rows()).map_collect(|r| r.sum());
        init_tprobs *= 1. / init_tprobs.sum();
        let mut init_tprobs_dip = &init_tprobs * &init_tprobs;
        init_tprobs_dip *= 1. / init_tprobs_dip.sum();
        let out = if let Some(rng) = rng {
            let ind = Self::sample(init_tprobs_dip.view(), rng);
            if save_tprobs_dip {
                (Some(ind), Some(init_tprobs_dip))
            } else {
                (Some(ind), None)
            }
        } else {
            (None, Some(init_tprobs_dip))
        };

        let mut _t = COMB_T.lock().unwrap();
        *_t += Instant::now() - t;
        out
    }

    fn combine_and_sample<R: Rng>(
        fprobs: ArrayView2<Real>,
        bprobs: ArrayView2<Real>,
        sampling_args: Option<(u8, R)>,
        save_tprobs_dip: bool,
    ) -> (Option<u8>, Option<Array2<Real>>) {
        let t = Instant::now();
        let out = if let Some((prev_ind, rng)) = sampling_args {
            if save_tprobs_dip {
                let mut tprobs = fprobs.dot(&bprobs.t());
                let ind = Self::sample_dip(
                    tprobs.row(prev_ind as usize),
                    tprobs.row(P - 1 - prev_ind as usize),
                    rng,
                );
                tprobs *= 1. / tprobs.sum();
                (Some(ind), Some(Self::combine_dip(tprobs.view())))
            } else {
                let tprobs_row_1 = fprobs.row(prev_ind as usize).dot(&bprobs.t());
                let tprobs_row_2 = fprobs.row(P - 1 - prev_ind as usize).dot(&bprobs.t());
                let ind = Self::sample_dip(tprobs_row_1.view(), tprobs_row_2.view(), rng);
                (Some(ind), None)
            }
        } else {
            let mut tprobs = fprobs.dot(&bprobs.t());
            tprobs *= 1. / tprobs.sum();
            (None, Some(Self::combine_dip(tprobs.view())))
        };

        let mut _t = COMB_T.lock().unwrap();
        *_t += Instant::now() - t;
        out
    }

    fn combine_dip(tprobs: ArrayView2<Real>) -> Array2<Real> {
        let mut tprobs_dip = Array2::from_shape_fn((P, P), |(i, j)| {
            tprobs[[i, j]] * tprobs[[P - i - 1, P - j - 1]]
        });
        tprobs_dip *= 1. / tprobs_dip.sum();
        tprobs_dip
    }

    fn sample_dip(
        tprobs1: ArrayView1<Real>,
        #[cfg(feature = "leak-resist-new")] tprobs1_e: ArrayView1<TpI16>,
        tprobs2: ArrayView1<Real>,
        #[cfg(feature = "leak-resist-new")] tprobs2_e: ArrayView1<TpI16>,
        rng: impl Rng,
    ) -> u8 {
        let tprobs1 = &tprobs1 * 1. / tprobs1.sum();
        let tprobs2 = &tprobs2 * 1. / tprobs2.sum();

        let tprobs_dip = Array1::<Real>::from_shape_fn(P, |i| tprobs1[i] * tprobs2[P - 1 - i]);

        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_dip_e =
            Array1::<TpI16>::from_shape_fn(P, |i| weights1_e[i] + weights2_e[P - 1 - i]);

        #[cfg(feature = "leak-resist-new")]
        renorm_equalize_scale_arr1(tprobs_dip.view_mut(), tprobs_dip_e.view_mut());

        Self::sample(tprobs_dip.view(), rng)
    }

    fn sample(tprobs_dip: ArrayView1<Real>, mut rng: impl Rng) -> u8 {
        let mut total_weight = tprobs_dip[0];
        let mut cumulative_weights: Vec<Real> = Vec::with_capacity(tprobs_dip.len());
        cumulative_weights.push(total_weight);
        for &w in tprobs_dip.iter().skip(1) {
            total_weight += w;
            cumulative_weights.push(total_weight);
        }

        let chosen_weight = tp_value_real!(rng.gen_range(0.0..1.0), f32) * total_weight;

        #[cfg(feature = "leak-resist-new")]
        {
            let mut index = TpU32::protect(0);
            let mut done = TpBool::protect(false);
            for w in cumulative_weights {
                done = w.tp_gt(&chosen_weight).select(TpBool::protect(true), done);
                index = (!done).select(index + 1, index);
            }
            index
                .tp_eq(&(weights.len() as u32))
                .select(TpU8::protect(0), index)
                .expose()
        }

        #[cfg(not(feature = "leak-resist-new"))]
        {
            use std::cmp::Ordering;
            cumulative_weights
                .binary_search_by(|w| {
                    if *w <= chosen_weight {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                })
                .unwrap_err() as u8
        }
    }
}

#[cfg(not(feature = "leak-resist-new"))]
pub struct Hmm {}

#[cfg(feature = "leak-resist-new")]
pub struct Hmm {
    pub bprobs_e: Array2<TpI16>,
    cur_fprobs_e: Array1<TpI16>,
    prev_fprobs_e: Array1<TpI16>,
    pub tprobs_e: Array3<TpI16>,
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
            tprobs_e: Array3::from_elem((0, 0, 0), TpI16::protect(0)),
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
        ignored_sites: ArrayView1<BoolMcc>,
        is_first_window: bool,
    ) -> Array3<Real> {
        let m = ref_panel.nrows();
        let k = ref_panel.ncols();
        let mut rprobs_iter = rprobs.get_forward();

        let bprobs = self.backward(ref_panel, genograph, hmm_params, rprobs, ignored_sites);

        #[cfg(feature = "leak-resist-new")]
        {
            self.is_backward = false;
        }

        let mut tprobs = Array3::<Real>::zeros((m, P, P));

        #[cfg(feature = "leak-resist-new")]
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

        #[cfg(feature = "leak-resist-new")]
        self.prev_fprobs_e.assign(&self.cur_fprobs_e);

        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_3x = Array3::<Real>::zeros(((m + 2) / 3, P, P));
        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_3x_e = Array3::<TpI16>::from_elem(((m + 2) / 3, P, P), TpI16::protect(0));
        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_3x_fwd = prev_fprobs.clone();
        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_3x_fwd_e = self.prev_fprobs_e.clone();
        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_3x_bwd = bprobs.slice(s![0, .., ..]).to_owned();
        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_3x_bwd_e = self.bprobs_e.row(0).to_owned();

        for i in 1..m {
            let rprobs = rprobs_iter.next().unwrap();

            #[cfg(not(feature = "leak-resist-new"))]
            if ignored_sites[i] {
                continue;
            }

            #[cfg(feature = "leak-resist-new")]
            {
                self.cur_i = i;
            }

            self.transition(rprobs, prev_fprobs.view(), cur_fprobs.view_mut());

            #[cfg(feature = "leak-resist-new")]
            {
                let cond = TpBool::protect(genograph[i].is_segment_marker());

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

            #[cfg(not(feature = "leak-resist-new"))]
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
                hmm_params,
                cur_fprobs.view_mut(),
            );

            #[cfg(feature = "leak-resist-new")]
            {
                let cond = !ignored_sites[i];
                Zip::from(&cur_fprobs)
                    .and(&mut prev_fprobs)
                    .for_each(|c, p| *p = cond.select(*c, *p));

                Zip::from(&self.cur_fprobs_e)
                    .and(&mut self.prev_fprobs_e)
                    .for_each(|c, p| *p = cond.select(*c, *p));
            }

            #[cfg(not(feature = "leak-resist-new"))]
            prev_fprobs.assign(&cur_fprobs);
        }

        #[cfg(feature = "leak-resist-new")]
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

    pub fn combine_dips(&self, tprobs: ArrayView2<Real>, tprobs_dips: ArrayViewMut2<f64>) {
        let t = Instant::now();

        #[cfg(not(feature = "leak-resist-new"))]
        let mut tprobs_dips = tprobs_dips;

        #[cfg(feature = "leak-resist-new")]
        let tprobs_e = self.tprobs_e.slice(s![self.cur_i, .., ..]);

        #[cfg(not(feature = "leak-resist-new"))]
        let tprobs = &tprobs * (1. / tprobs.sum());

        #[cfg(feature = "leak-resist-new")]
        let mut _tprobs_dips = tprobs_dips;

        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_dips = Array2::<Real>::zeros((P, P));

        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_dips_e_ext = Array2::<TpI16>::from_elem((P, P), TpI16::protect(0));

        for i in 0..P {
            for j in 0..P {
                tprobs_dips[[i, j]] = tprobs[[i, j]] * tprobs[[P - 1 - i, P - 1 - j]];
                #[cfg(feature = "leak-resist-new")]
                {
                    tprobs_dips_e_ext[[i, j]] = tprobs_e[[i, j]] + tprobs_e[[P - 1 - i, P - 1 - j]];
                }
            }
        }

        #[cfg(feature = "leak-resist-new")]
        let mut tprobs_dips_e = {
            let mut tprobs_dips_e = Array1::<TpI16>::from_elem(P, TpI16::protect(0));
            renorm_equalize_scale(
                tprobs_dips.view_mut(),
                tprobs_dips_e_ext.view_mut(),
                tprobs_dips_e.view_mut(),
            );
            tprobs_dips_e
        };

        #[cfg(feature = "leak-resist-new")]
        let (sum, sum_e) = sum_scale(tprobs_dips.view(), tprobs_dips_e.view());

        #[cfg(not(feature = "leak-resist-new"))]
        let sum = tprobs_dips.sum();

        tprobs_dips *= tp_value_real!(1, i64) / sum;

        #[cfg(feature = "leak-resist-new")]
        {
            tprobs_dips_e.iter_mut().for_each(|v| *v -= sum_e);
            renorm_scale(tprobs_dips.view_mut(), tprobs_dips_e.view_mut());
            _tprobs_dips.assign(&debug_expose_array(
                tprobs_dips.view(),
                tprobs_dips_e.view(),
            ));
        }

        let mut _t = COMBD_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn backward(
        &mut self,
        ref_panel: ArrayView2<Genotype>,
        genograph: ArrayView1<G>,
        hmm_params: &HmmParams,
        rprobs: &RprobsSlice,
        ignored_sites: ArrayView1<BoolMcc>,
    ) -> Array3<Real> {
        let m = ref_panel.nrows();
        let n = ref_panel.ncols();
        let mut rprobs_iter = rprobs.get_backward();

        let mut bprobs = Array3::<Real>::zeros((m, P, n));

        #[cfg(feature = "leak-resist-new")]
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

            #[cfg(not(feature = "leak-resist-new"))]
            if ignored_sites[i] && !genograph[i + 1].is_segment_marker() {
                cur_bprob.assign(&prev_bprob);
                continue;
            }

            #[cfg(feature = "leak-resist-new")]
            {
                self.cur_i = i;
            }

            self.transition(rprobs, prev_bprob.view(), cur_bprob.view_mut());

            #[cfg(feature = "leak-resist-new")]
            {
                let cond = TpBool::protect(genograph[i + 1].is_segment_marker());
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

            #[cfg(not(feature = "leak-resist-new"))]
            if genograph[i + 1].is_segment_marker() {
                self.collapse(cur_bprob.view_mut());
            }

            self.emission(
                ref_panel.row(i),
                genograph[i],
                hmm_params,
                cur_bprob.view_mut(),
            );

            #[cfg(feature = "leak-resist-new")]
            {
                let cond =
                    !ignored_sites[i] | TpBool::protect(genograph[i + 1].is_segment_marker());
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
                #[cfg(feature = "leak-resist-new")]
                {
                    let z = TpI8::protect(z);
                    let g = TpI8::protect(graph_col.get_row(i));
                    *p = (z.tp_not_eq(&g)).select(hmm_params.eprob, Real::protect_i64(1));
                }

                #[cfg(not(feature = "leak-resist-new"))]
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
        mut probs: ArrayViewMut2<Real>,
    ) {
        let t = Instant::now();

        Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
            Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
                #[cfg(feature = "leak-resist-new")]
                {
                    let z = TpI8::protect(z);
                    let g = TpI8::protect(graph_col.get_row(i));
                    //*p = (z.tp_not_eq(&g)).select(*p * hmm_params.eprob, *p);
                    *p = (z.tp_not_eq(&g))
                        .select((*p >> 14) + (*p >> 15) + (*p >> 17) + (*p >> 20), *p);
                }

                #[cfg(not(feature = "leak-resist-new"))]
                if z != graph_col.get_row(i) {
                    *p *= hmm_params.eprob;
                }
            });
        });

        #[cfg(feature = "leak-resist-new")]
        renorm_scale(probs, self.get_cur_probs_e());

        let mut _t = EMIS_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn transition(
        &mut self,
        rprob: (Real, Real),
        prev_probs: ArrayView2<Real>,
        mut cur_probs: ArrayViewMut2<Real>,
    ) {
        let t = Instant::now();
        #[cfg(feature = "leak-resist-new")]
        let (mut cur_probs_e, prev_probs_e) = self.get_probs_e();

        //#[cfg(feature = "leak-resist-new")]
        //debug_sum_by_row(prev_probs, prev_probs_e.view());

        let all_sum_h = Zip::from(prev_probs.rows()).map_collect(|r| r.sum());

        #[cfg(feature = "leak-resist-new")]
        let all_sum_h_e = prev_probs_e.to_owned();

        let rprob = rprob.0 / rprob.1;

        let _all_sum_h = &all_sum_h * rprob;

        Zip::from(cur_probs.rows_mut())
            .and(prev_probs.rows())
            .and(&_all_sum_h)
            .for_each(|mut cr, pr, &s| cr.assign(&(&pr + s)));

        #[cfg(feature = "leak-resist-new")]
        {
            cur_probs_e.assign(&all_sum_h_e);
            renorm_scale(cur_probs.view_mut(), cur_probs_e.view_mut());
        }

        //#[cfg(not(feature = "leak-resist-new"))]
        //{
        //let sum: Real = all_sum_h.sum();
        //let scale = tp_value_real!(1, i64) / sum;
        //cur_probs *= scale;
        //}
        #[cfg(not(feature = "leak-resist-new"))]
        {
            let sum = cur_probs.sum();
            if sum < 1e-10 || sum > 1e10 {
                cur_probs *= 1. / cur_probs.sum();
            }
        }

        let mut _t = TRAN_T.lock().unwrap();
        *_t += Instant::now() - t;
    }

    fn collapse(&mut self, mut cur_probs: ArrayViewMut2<Real>) {
        let t = Instant::now();

        #[cfg(feature = "leak-resist-new")]
        let (sum_k, sum_k_e) = sum_scale_by_column(cur_probs.view(), self.get_cur_probs_e().view());

        #[cfg(not(feature = "leak-resist-new"))]
        let mut sum_k = Zip::from(cur_probs.columns()).map_collect(|c| c.sum());

        #[cfg(not(feature = "leak-resist-new"))]
        {
            let sum = sum_k.sum();
            let scale = tp_value_real!(1, i64) / sum;
            sum_k *= scale;
        }

        #[cfg(feature = "leak-resist-new")]
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
        //#[cfg(feature = "leak-resist-new")]
        //debug_sum_by_row(first_bprobs, self.bprobs_e.row(0));

        let init_probs = Zip::from(first_bprobs.rows()).map_collect(|r| r.sum());

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

    #[cfg(feature = "leak-resist-new")]
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

    #[cfg(not(feature = "leak-resist-new"))]
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
        probs: ArrayView2<Real>,
        probs_e: ArrayView1<TpI16>,
    ) -> (Array1<Real>, Array1<TpI16>) {
        //debug_sum_by_row(probs, probs_e);
        let mut sum_by_row = Zip::from(probs.rows()).map_collect(|r| r.sum());
        let mut sum_by_row_e = probs_e.to_owned();
        renorm_scale_arr1(sum_by_row.view_mut(), sum_by_row_e.view_mut());
        (sum_by_row, sum_by_row_e)
    }

    pub fn debug_sum_by_row(probs: ArrayView2<Real>, probs_e: ArrayView1<TpI16>) {
        Zip::from(probs.rows()).and(&probs_e).for_each(|r, &e| {
            if r.iter()
                .map(|v| v.into_inner().expose() as f64)
                .sum::<f64>()
                > i64::MAX as f64
            {
                println!("{:#?}", debug_expose_row(r, e));
                println!("{:#?}", e.expose());
                println!("{:#?}", debug_expose_s_arr1(r));
            }
        });
    }

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

    pub fn renorm_equalize_scale_arr1<const F: usize>(
        probs: ArrayViewMut1<TpFixed64<F>>,
        probs_e: ArrayViewMut1<TpI16>,
    ) -> TpI16 {
        let e_to_match =
            Zip::from(&probs)
                .and(&probs_e)
                .fold(TpI16::protect(i16::MIN), |accu, p, &e| {
                    let new_e =
                        TpI16::protect(64) - p.leading_zeros().as_i16() - TpI16::protect(F as i16)
                            + e;
                    accu.tp_gt(&new_e).select(accu, new_e)
                });
        match_scale_arr1(e_to_match, probs, probs_e);
        e_to_match
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
}
