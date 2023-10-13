mod params;
mod sampling;
mod viterbi;
mod windows_split;
pub use params::*;

use crate::genotype_graph::GenotypeGraph;
use crate::hmm::{combine_dips, Hmm};
use crate::hmm::{COLL_T, COMBD_T, COMB_T, EMIS_T, TRAN_T};
use crate::variants::{Rarity, Variant};
use crate::{tp_value, Bool, Genotype, Real, Usize, U8};
use common::ref_panel::RefPanelSlice;
use rand::Rng;

use std::time::{Duration, Instant};

#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::{TpEq, TpI16, TpI8, TpOrd};

use ndarray::{s, Array1, Array2, Array3, ArrayView1, Zip};

const N_HETS_PER_SEGMENT: usize = 3;
const P: usize = 1 << N_HETS_PER_SEGMENT;

#[derive(PartialEq, Eq, Clone)]
pub enum IterOption {
    Burnin(usize),
    Pruning(usize),
    Main(usize),
}

#[derive(PartialEq, Eq, Debug, Clone)]
enum IterOptionInternal {
    Burnin,
    Pruning,
    Main(bool), // First main iteration?
}

pub struct Mcmc<'a> {
    params: &'a McmcSharedParams,
    cur_overlap_region_len: usize,
    genotype_graph: GenotypeGraph,
    ignored_sites: Array1<Bool>,
    estimated_haps: Array2<Genotype>,
    phased_ind: Array2<U8>,
    tprobs: Array3<Real>,
    #[cfg(feature = "obliv")]
    tprobs_e: Array2<TpI8>,
}

impl<'a> Mcmc<'a> {
    pub fn run(
        params: &'a McmcSharedParams,
        genotypes: ArrayView1<Genotype>,
        iterations: &[IterOption],
        mut rng: impl Rng,
    ) -> Array2<Genotype> {
        match iterations.last().unwrap() {
            IterOption::Main(_) => {}
            _ => panic!("Last iterations must be main."),
        }

        let mut iterations_iternal = Vec::new();
        for i in iterations {
            match i {
                IterOption::Burnin(r) => {
                    assert!(*r > 0);
                    for _ in 0..*r {
                        iterations_iternal.push(IterOptionInternal::Burnin);
                    }
                }
                IterOption::Pruning(r) => {
                    assert!(*r > 0);
                    for _ in 0..*r {
                        iterations_iternal.push(IterOptionInternal::Pruning);
                    }
                }
                IterOption::Main(r) => {
                    assert!(*r > 0);
                    iterations_iternal.push(IterOptionInternal::Main(true));
                    for _ in 1..*r {
                        iterations_iternal.push(IterOptionInternal::Main(false));
                    }
                }
            }
        }

        let mut mcmc = Self::initialize(&params, genotypes.view());

        for i in iterations_iternal {
            //mcmc.iteration(i, &mut rng);
            mcmc.iteration_rss(i, &mut rng);
        }

        mcmc.phased_ind = viterbi::viterbi(mcmc.tprobs.view(), mcmc.genotype_graph.graph.view());

        mcmc.genotype_graph
            .traverse_graph_pair(mcmc.phased_ind.view(), mcmc.estimated_haps.view_mut());

        mcmc.estimated_haps
    }

    fn initialize(params: &'a McmcSharedParams, genotypes: ArrayView1<Genotype>) -> Self {
        println!("=== Initialization ===",);
        let now = Instant::now();
        #[cfg(feature = "obliv")]
        use compressed_pbwt_obliv::mcmc_init::mcmc_init;

        #[cfg(not(feature = "obliv"))]
        use compressed_pbwt::mcmc_init::mcmc_init;

        let (h_0, h_1) = mcmc_init(
            genotypes.as_slice().unwrap(),
            &params.pbwt_tries,
            params.ref_panel.n_haps,
        );

        #[cfg(feature = "obliv")]
        let estimated_haps = Array2::from_shape_fn((genotypes.len(), 2), |(i, j)| match j {
            0 => h_0[i].as_i8(),
            1 => h_1[i].as_i8(),
            _ => panic!(),
        });
        #[cfg(not(feature = "obliv"))]
        let estimated_haps = Array2::from_shape_fn((genotypes.len(), 2), |(i, j)| match j {
            0 => h_0[i] as i8,
            1 => h_1[i] as i8,
            _ => panic!(),
        });

        #[cfg(feature = "obliv")]
        let phased_ind = Array2::<U8>::from_elem((estimated_haps.nrows(), 2), U8::protect(0));

        #[cfg(not(feature = "obliv"))]
        let phased_ind = Array2::<U8>::zeros((estimated_haps.nrows(), 2));
        let tprobs = Array3::<Real>::zeros((estimated_haps.nrows(), P, P));

        #[cfg(feature = "obliv")]
        let tprobs_e = Array2::<TpI8>::from_elem((estimated_haps.nrows(), P), TpI8::protect(0));

        let genotype_graph = GenotypeGraph::build(genotypes);
        let ignored_sites = Self::get_ignored_sites(params.variants.view(), genotypes);
        println!("Initialization: {} ms", (Instant::now() - now).as_millis());
        println!("",);
        Self {
            params,
            cur_overlap_region_len: params.overlap_region_len,
            genotype_graph,
            estimated_haps,
            phased_ind,
            tprobs,
            ignored_sites,
            #[cfg(feature = "obliv")]
            tprobs_e,
        }
    }

    fn iteration(&mut self, iter_option: IterOptionInternal, mut rng: impl Rng) {
        {
            *EMIS_T.lock().unwrap() = Duration::from_millis(0);
            *TRAN_T.lock().unwrap() = Duration::from_millis(0);
            *COLL_T.lock().unwrap() = Duration::from_millis(0);
            *COMB_T.lock().unwrap() = Duration::from_millis(0);
            *COMBD_T.lock().unwrap() = Duration::from_millis(0);
        }

        println!("=== {:?} Iteration ===", iter_option);
        let now = Instant::now();

        let mut pbwt_group_filter = self.params.randomize_pbwt_group_bitmask(&mut rng);

        pbwt_group_filter
            .iter_mut()
            .zip(&self.params.pbwt_evaluted)
            .for_each(|(a, &b)| *a = *a && b);

        let neighbors = {
            #[cfg(feature = "obliv")]
            let (h_0, h_1): (Vec<_>, Vec<_>) = self
                .estimated_haps
                .rows()
                .into_iter()
                .map(|v| (v[0].tp_eq(&1), v[1].tp_eq(&1)))
                .unzip();

            #[cfg(not(feature = "obliv"))]
            let (h_0, h_1): (Vec<_>, Vec<_>) = self
                .estimated_haps
                .rows()
                .into_iter()
                .map(|v| (v[0] == 1, v[1] == 1))
                .unzip();

            #[cfg(feature = "obliv")]
            use compressed_pbwt_obliv::nn::find_top_neighbors;

            #[cfg(not(feature = "obliv"))]
            use compressed_pbwt::nn::find_top_neighbors;

            let mut nn_0 = find_top_neighbors(
                &h_0,
                self.params.s,
                &self.params.pbwt_tries,
                self.params.ref_panel.n_haps,
                &pbwt_group_filter,
            );

            let nn_1 = find_top_neighbors(
                &h_1,
                self.params.s,
                &self.params.pbwt_tries,
                self.params.ref_panel.n_haps,
                &pbwt_group_filter,
            );

            for (a, b) in nn_0.iter_mut().zip(nn_1.into_iter()) {
                a.as_mut().map(|v| v.extend(b.unwrap().into_iter()));
            }
            nn_0
        };

        let windows = self.windows(&mut rng);

        #[cfg(feature = "obliv")]
        let mut prev_ind = (U8::protect(0), U8::protect(0));
        #[cfg(not(feature = "obliv"))]
        let mut prev_ind = (0, 0);
        let mut sum_window_size = 0;
        let n_windows = windows.len();
        let mut ks = Vec::new();
        let mut tprob_pairs = Array2::<Real>::zeros((P, P));

        let mut i = 0;
        let mut is_first_window = true;

        let rprobs = self.params.hmm_params.get_rprobs(
            self.ignored_sites.view(),
            &self.genotype_graph,
            self.params.variants.view(),
        );

        for ((start_w, end_w), (start_write_w, end_write_w)) in windows.into_iter() {
            sum_window_size +=
                self.params.variants[end_w - 1].bp - self.params.variants[start_w].bp;
            let genotype_graph_w = self.genotype_graph.slice(start_w, end_w);
            let params_w = self.params.slice(start_w, end_w);
            let rprobs_w = rprobs.slice(start_w, end_w);
            let neighbors_w = &neighbors[start_w..end_w];
            let ignored_sites_w = self.ignored_sites.slice(s![start_w..end_w]);
            let (selected_ref_panel, k) = select_ref_panel(&params_w.ref_panel, neighbors_w);
            ks.push(k as f64);

            let mut hmm = Hmm::new();
            let tprobs_window = hmm.forward_backward(
                selected_ref_panel.view(),
                genotype_graph_w.graph.view(),
                &self.params.hmm_params,
                &rprobs_w,
                ignored_sites_w,
                is_first_window,
            );

            #[cfg(not(feature = "obliv"))]
            let mut tprobs_window = tprobs_window;

            #[cfg(feature = "obliv")]
            let tprobs_window_src = tprobs_window.view();

            #[cfg(feature = "obliv")]
            let tprobs_window_src_e = hmm.tprobs_e.view();

            #[cfg(feature = "obliv")]
            let mut tprobs_window_target = self.tprobs.slice_mut(s![start_w..end_w, .., ..]);

            #[cfg(not(feature = "obliv"))]
            let tprobs_window_src =
                tprobs_window.slice_mut(s![start_write_w - start_w..end_write_w - start_w, .., ..]);

            #[cfg(not(feature = "obliv"))]
            let genotype_graph = &self.genotype_graph;

            #[cfg(not(feature = "obliv"))]
            let mut tprobs_window_target =
                self.tprobs
                    .slice_mut(s![start_write_w..end_write_w, .., ..]);

            #[cfg(feature = "obliv")]
            let mut j = 0u64;

            if iter_option == IterOptionInternal::Pruning
                || iter_option == IterOptionInternal::Main(true)
            {
                Zip::from(tprobs_window_target.outer_iter_mut())
                    .and(tprobs_window_src.outer_iter())
                    .for_each(|t, s| {
                        #[cfg(feature = "obliv")]
                        {
                            let cond = (start_write_w - start_w as u64).tp_lt_eq(&j)
                                & (end_write_w - start_w as u64).tp_gt(&j);
                            let s_e = tprobs_window_src_e.slice(s![j as usize, .., ..]);
                            combine_dips(s, s_e, t, cond);
                            j += 1;
                        }

                        #[cfg(not(feature = "obliv"))]
                        if i == 0 || genotype_graph.graph[i].is_segment_marker() {
                            combine_dips(s, t);
                        } else {
                            let mut t = t;
                            t.fill(0.);
                        }
                        i += 1;
                    });
            } else if iter_option == IterOptionInternal::Main(false) {
                Zip::from(tprobs_window_target.outer_iter_mut())
                    .and(tprobs_window_src.outer_iter())
                    .for_each(|mut t, s| {
                        #[cfg(feature = "obliv")]
                        {
                            let cond = (start_write_w - start_w as u64).tp_lt_eq(&j)
                                & (end_write_w - start_w as u64).tp_gt(&j);
                            tprob_pairs.fill(Real::ZERO);
                            let s_e = tprobs_window_src_e.slice(s![j as usize, .., ..]);
                            combine_dips(s, s_e, tprob_pairs.view_mut(), cond);
                            t += &tprob_pairs;
                            j += 1;
                        }

                        #[cfg(not(feature = "obliv"))]
                        if i == 0 || genotype_graph.graph[i].is_segment_marker() {
                            combine_dips(s, tprob_pairs.view_mut());
                            t += &tprob_pairs;
                        }

                        i += 1;
                    });
            }

            #[cfg(not(feature = "obliv"))]
            let tprobs_window_src =
                tprobs_window.slice_mut(s![start_write_w - start_w..end_write_w - start_w, .., ..]);

            // sample
            #[cfg(feature = "obliv")]
            let phased_ind_window = sampling::forward_sampling(
                prev_ind,
                tprobs_window.view(),
                hmm.tprobs_e.view(),
                genotype_graph_w.graph.view(),
                is_first_window,
                start_write_w - start_w as u64,
                &mut rng,
            );

            #[cfg(not(feature = "obliv"))]
            let phased_ind_window = sampling::forward_sampling(
                prev_ind,
                tprobs_window_src.view(),
                genotype_graph_w
                    .graph
                    .slice(s![start_write_w - start_w..end_write_w - start_w]),
                is_first_window,
                &mut rng,
            );

            #[cfg(feature = "obliv")]
            let mut tmp = self.phased_ind.slice_mut(s![start_w..end_w, ..]);

            #[cfg(not(feature = "obliv"))]
            let mut tmp = self
                .phased_ind
                .slice_mut(s![start_write_w..end_write_w, ..]);

            #[cfg(feature = "obliv")]
            {
                ndarray::Zip::indexed(tmp.rows_mut())
                    .and(phased_ind_window.rows())
                    .for_each(|i, mut t, s| {
                        let cond = (start_write_w - start_w as u64).tp_lt_eq(&(i as u64))
                            & (end_write_w - start_w as u64).tp_gt(&(i as u64));
                        t[0] = cond.select(s[0], t[0]);
                        t[1] = cond.select(s[1], t[1]);
                    });

                for i in 0..phased_ind_window.nrows() {
                    let cond = (end_write_w - start_w as u64 - 1).tp_eq(&(i as u64));
                    prev_ind.0 = cond.select(phased_ind_window[[i, 0]], prev_ind.0);
                    prev_ind.1 = cond.select(phased_ind_window[[i, 1]], prev_ind.1);
                }
            }
            #[cfg(not(feature = "obliv"))]
            {
                tmp.assign(&phased_ind_window);
                prev_ind = (
                    phased_ind_window[[phased_ind_window.nrows() - 1, 0]],
                    phased_ind_window[[phased_ind_window.nrows() - 1, 1]],
                );
            }
            is_first_window = false;
        }

        self.genotype_graph
            .traverse_graph_pair(self.phased_ind.view(), self.estimated_haps.view_mut());

        if iter_option == IterOptionInternal::Pruning {
            self.genotype_graph.prune(self.tprobs.view());
            self.cur_overlap_region_len *= 2;
        }

        #[cfg(not(feature = "obliv"))]
        println!(
            "#Segments: {}",
            self.genotype_graph
                .graph
                .iter()
                .filter(|g| g.is_segment_marker())
                .count()
        );
        use statrs::statistics::Statistics;
        println!(
            "K: {:.3}+/-{:.3}",
            Statistics::mean(&ks),
            Statistics::std_dev(&ks)
        );

        println!(
            "Window size: {:.2} Mb",
            sum_window_size as f64 / n_windows as f64 / 1e6
        );

        println!("Elapsed time: {} ms", (Instant::now() - now).as_millis());
        println!("Emission: {} ms", EMIS_T.lock().unwrap().as_millis());
        println!("Transition: {} ms", TRAN_T.lock().unwrap().as_millis());
        println!("Collapse: {} ms", COLL_T.lock().unwrap().as_millis());
        println!("Combine: {} ms", COMB_T.lock().unwrap().as_millis());
        println!(
            "Combine Diploid: {} ms",
            COMBD_T.lock().unwrap().as_millis()
        );
        println!("",);
    }

    fn iteration_rss(&mut self, iter_option: IterOptionInternal, mut rng: impl Rng) {
        println!("=== {:?} Iteration ===", iter_option);
        let now = Instant::now();

        let mut pbwt_group_filter = self.params.randomize_pbwt_group_bitmask(&mut rng);
        pbwt_group_filter
            .iter_mut()
            .zip(&self.params.pbwt_evaluted)
            .for_each(|(a, &b)| *a = *a && b);

        let neighbors = {
            #[cfg(feature = "obliv")]
            let (h_0, h_1): (Vec<_>, Vec<_>) = self
                .estimated_haps
                .rows()
                .into_iter()
                .map(|v| (v[0].tp_eq(&1), v[1].tp_eq(&1)))
                .unzip();

            #[cfg(not(feature = "obliv"))]
            let (h_0, h_1): (Vec<_>, Vec<_>) = self
                .estimated_haps
                .rows()
                .into_iter()
                .map(|v| (v[0] == 1, v[1] == 1))
                .unzip();

            #[cfg(feature = "obliv")]
            use compressed_pbwt_obliv::nn::find_top_neighbors;

            #[cfg(not(feature = "obliv"))]
            use compressed_pbwt::nn::find_top_neighbors;

            let mut nn_0 = find_top_neighbors(
                &h_0,
                self.params.s,
                &self.params.pbwt_tries,
                self.params.ref_panel.n_haps,
                &pbwt_group_filter,
            );

            let nn_1 = find_top_neighbors(
                &h_1,
                self.params.s,
                &self.params.pbwt_tries,
                self.params.ref_panel.n_haps,
                &pbwt_group_filter,
            );

            for (a, b) in nn_0.iter_mut().zip(nn_1.into_iter()) {
                a.as_mut().map(|v| v.extend(b.unwrap().into_iter()));
            }
            nn_0
        };

        let windows = self.windows(&mut rng);

        #[cfg(feature = "obliv")]
        let mut prev_ind = (U8::protect(0), U8::protect(0));
        #[cfg(not(feature = "obliv"))]
        let mut prev_ind = (0, 0);
        let mut sum_window_size = 0;
        let n_windows = windows.len();
        let mut ks = Vec::new();
        let mut tprob_pairs = Array2::<Real>::zeros((P, P));

        let mut i = 0;
        let mut is_first_window = true;

        let rprobs = self.params.hmm_params.get_rprobs(
            //self.ignored_sites.view(),
            Array1::from_elem(self.ignored_sites.len(), tp_value!(false, bool)).view(),
            &self.genotype_graph,
            self.params.variants.view(),
        );

        for ((start_w, end_w), (start_write_w, end_write_w)) in windows.into_iter() {
            sum_window_size +=
                self.params.variants[end_w - 1].bp - self.params.variants[start_w].bp;
            let genotype_graph_w = self.genotype_graph.slice(start_w, end_w);
            let params_w = self.params.slice(start_w, end_w);
            let rprobs_w = rprobs.slice(start_w, end_w);
            let neighbors_w = &neighbors[start_w..end_w];
            let ignored_sites_w = self.ignored_sites.slice(s![start_w..end_w]);
            let (full_filter, k) = find_nn_bitmap(neighbors_w, params_w.ref_panel.n_haps);
            ks.push(k as f64);

            let filtered_blocks = params_w
                .ref_panel
                .blocks
                .iter()
                .map(|b| {
                    crate::rss_hmm::filtered_block::FilteredBlockSlice::from_block_slice(
                        b,
                        &full_filter,
                    )
                })
                .collect::<Vec<_>>();

            let fwbw_out = crate::rss_hmm::HmmReduced::fwbw(
                &filtered_blocks,
                params_w.ref_panel.n_sites,
                genotype_graph_w.graph.view(),
                self.params.hmm_params.eprob,
                &rprobs_w,
                //ignored_sites_w,
                Array1::from_elem(ignored_sites_w.len(), tp_value!(false, bool)).view(),
            );
            #[cfg(feature = "obliv")]
            let (first_tprobs, first_tprobs_e, tprobs, tprobs_e) = fwbw_out;

            #[cfg(not(feature = "obliv"))]
            let (first_tprobs, tprobs) = fwbw_out;

            let mut tprobs_window = Array3::<Real>::zeros(((end_w - start_w), P, P));

            #[cfg(feature = "obliv")]
            let mut tprobs_window_e =
                Array3::<TpI16>::from_elem(((end_w - start_w), P, P), TpI16::protect(0));

            let mut j = 0;
            Zip::indexed(tprobs_window.outer_iter_mut())
                .and(&genotype_graph_w.graph)
                .for_each(|i, mut t, g| {
                    #[cfg(feature = "obliv")]
                    if g.is_segment_marker().expose() {
                        t.assign(&tprobs.slice(s![j, .., ..]));
                        tprobs_window_e
                            .slice_mut(s![i, .., ..])
                            .assign(&tprobs_e.slice(s![j, .., ..]));
                        j += 1;
                    }

                    #[cfg(not(feature = "obliv"))]
                    if g.is_segment_marker() {
                        t.assign(&tprobs.slice(s![j, .., ..]));
                        j += 1;
                    }
                });

            if is_first_window {
                Zip::from(tprobs_window.slice_mut(s![0, .., ..]).rows_mut())
                    .for_each(|mut r| r.assign(&first_tprobs));

                #[cfg(feature = "obliv")]
                Zip::from(tprobs_window_e.slice_mut(s![0, .., ..]).rows_mut())
                    .for_each(|mut r| r.assign(&first_tprobs_e));
            }

            //{
            //let (selected_ref_panel, k) = select_ref_panel(&params_w.ref_panel, neighbors_w);
            //let mut hmm = Hmm::new();
            //let tprobs_window_ref = hmm.forward_backward(
            //selected_ref_panel.view(),
            //genotype_graph_w.graph.view(),
            //&self.params.hmm_params,
            //&rprobs_w,
            ////ignored_sites_w,
            //Array1::from_elem(ignored_sites_w.len(), tp_value!(false, bool)).view(),
            //is_first_window,
            //);

            //println!("window {:?}", (start_w, end_w));
            //Zip::indexed(tprobs_window_ref.outer_iter())
            //.and(tprobs_window.outer_iter())
            //.and(&genotype_graph_w.graph)
            //.for_each(|i, r, t, g| {
            //#[cfg(feature = "obliv")]
            //let cond = g.is_segment_marker().expose();

            //#[cfg(not(feature = "obliv"))]
            //let cond = g.is_segment_marker();

            //if cond {
            //#[cfg(feature = "obliv")]
            //let r = crate::dynamic_fixed::debug_expose_array_ext(
            //r,
            //hmm.tprobs_e.slice(s![i, .., ..]),
            //);
            //#[cfg(feature = "obliv")]
            //let t = crate::dynamic_fixed::debug_expose_array_ext(
            //t,
            //tprobs_window_e.slice(s![i, .., ..]),
            //);

            //let ref_normalized = &r / r.sum();
            //let test_normalized = &t / t.sum();
            //if !ref_normalized.relative_eq(&test_normalized, 1e-6, 0.01) {
            ////if !ref_normalized.abs_diff_eq(&test_normalized, f64::EPSILON) {
            //println!("site {i}");
            //println!("ref:\n {:?}", r);
            //println!("test:\n {:?}", t);
            //println!("## normalized ##");
            //println!("ref:\n {:?}", ref_normalized);
            //println!("test:\n {:?}", test_normalized);
            //Zip::from(&ref_normalized).and(&test_normalized).for_each(
            //|a, b| {
            //if (a - b).abs() > 1e-6 {
            //println!("a: {}, b: {}", a, b);
            //}
            //},
            //);
            //panic!();
            //} else {
            ////println!("site {i} (OK)");
            //}
            //}
            //});
            //}

            #[cfg(feature = "obliv")]
            let tprobs_window_src = tprobs_window.view();

            #[cfg(feature = "obliv")]
            let tprobs_window_src_e = tprobs_window_e.view();

            #[cfg(feature = "obliv")]
            let mut tprobs_window_target = self.tprobs.slice_mut(s![start_w..end_w, .., ..]);

            #[cfg(not(feature = "obliv"))]
            let tprobs_window_src =
                tprobs_window.slice_mut(s![start_write_w - start_w..end_write_w - start_w, .., ..]);

            #[cfg(not(feature = "obliv"))]
            let genotype_graph = &self.genotype_graph;

            #[cfg(not(feature = "obliv"))]
            let mut tprobs_window_target =
                self.tprobs
                    .slice_mut(s![start_write_w..end_write_w, .., ..]);

            #[cfg(feature = "obliv")]
            let mut j = 0u64;

            if iter_option == IterOptionInternal::Pruning
                || iter_option == IterOptionInternal::Main(true)
            {
                Zip::from(tprobs_window_target.outer_iter_mut())
                    .and(tprobs_window_src.outer_iter())
                    .for_each(|t, s| {
                        #[cfg(feature = "obliv")]
                        {
                            let cond = (start_write_w - start_w as u64).tp_lt_eq(&j)
                                & (end_write_w - start_w as u64).tp_gt(&j);
                            let s_e = tprobs_window_src_e.slice(s![j as usize, .., ..]);
                            combine_dips(s, s_e, t, cond);
                            j += 1;
                        }

                        #[cfg(not(feature = "obliv"))]
                        if i == 0 || genotype_graph.graph[i].is_segment_marker() {
                            combine_dips(s, t);
                        } else {
                            let mut t = t;
                            t.fill(0.);
                        }
                        i += 1;
                    });
            } else if iter_option == IterOptionInternal::Main(false) {
                Zip::from(tprobs_window_target.outer_iter_mut())
                    .and(tprobs_window_src.outer_iter())
                    .for_each(|mut t, s| {
                        #[cfg(feature = "obliv")]
                        {
                            let cond = (start_write_w - start_w as u64).tp_lt_eq(&j)
                                & (end_write_w - start_w as u64).tp_gt(&j);
                            tprob_pairs.fill(Real::ZERO);
                            let s_e = tprobs_window_src_e.slice(s![j as usize, .., ..]);
                            combine_dips(s, s_e, tprob_pairs.view_mut(), cond);
                            t += &tprob_pairs;
                            j += 1;
                        }

                        #[cfg(not(feature = "obliv"))]
                        if i == 0 || genotype_graph.graph[i].is_segment_marker() {
                            combine_dips(s, tprob_pairs.view_mut());
                            t += &tprob_pairs;
                        }

                        i += 1;
                    });
            }

            #[cfg(not(feature = "obliv"))]
            let tprobs_window_src =
                tprobs_window.slice_mut(s![start_write_w - start_w..end_write_w - start_w, .., ..]);

            // sample
            #[cfg(feature = "obliv")]
            let phased_ind_window = sampling::forward_sampling(
                prev_ind,
                tprobs_window_src.view(),
                tprobs_window_src_e.view(),
                genotype_graph_w.graph.view(),
                is_first_window,
                start_write_w - start_w as u64,
                &mut rng,
            );

            #[cfg(not(feature = "obliv"))]
            let phased_ind_window = sampling::forward_sampling(
                prev_ind,
                tprobs_window_src.view(),
                genotype_graph_w
                    .graph
                    .slice(s![start_write_w - start_w..end_write_w - start_w]),
                is_first_window,
                &mut rng,
            );

            #[cfg(feature = "obliv")]
            let mut tmp = self.phased_ind.slice_mut(s![start_w..end_w, ..]);

            #[cfg(not(feature = "obliv"))]
            let mut tmp = self
                .phased_ind
                .slice_mut(s![start_write_w..end_write_w, ..]);

            #[cfg(feature = "obliv")]
            {
                ndarray::Zip::indexed(tmp.rows_mut())
                    .and(phased_ind_window.rows())
                    .for_each(|i, mut t, s| {
                        let cond = (start_write_w - start_w as u64).tp_lt_eq(&(i as u64))
                            & (end_write_w - start_w as u64).tp_gt(&(i as u64));
                        t[0] = cond.select(s[0], t[0]);
                        t[1] = cond.select(s[1], t[1]);
                    });

                for i in 0..phased_ind_window.nrows() {
                    let cond = (end_write_w - start_w as u64 - 1).tp_eq(&(i as u64));
                    prev_ind.0 = cond.select(phased_ind_window[[i, 0]], prev_ind.0);
                    prev_ind.1 = cond.select(phased_ind_window[[i, 1]], prev_ind.1);
                }
            }
            #[cfg(not(feature = "obliv"))]
            {
                tmp.assign(&phased_ind_window);
                prev_ind = (
                    phased_ind_window[[phased_ind_window.nrows() - 1, 0]],
                    phased_ind_window[[phased_ind_window.nrows() - 1, 1]],
                );
            }
            is_first_window = false;
        }

        self.genotype_graph
            .traverse_graph_pair(self.phased_ind.view(), self.estimated_haps.view_mut());

        if iter_option == IterOptionInternal::Pruning {
            self.genotype_graph.prune(self.tprobs.view());
            self.cur_overlap_region_len *= 2;
        }

        #[cfg(not(feature = "obliv"))]
        println!(
            "#Segments: {}",
            self.genotype_graph
                .graph
                .iter()
                .filter(|g| g.is_segment_marker())
                .count()
        );
        use statrs::statistics::Statistics;
        println!(
            "K: {:.3}+/-{:.3}",
            Statistics::mean(&ks),
            Statistics::std_dev(&ks)
        );

        println!(
            "Window size: {:.2} Mb",
            sum_window_size as f64 / n_windows as f64 / 1e6
        );

        println!("Elapsed time: {} ms", (Instant::now() - now).as_millis());
        println!("Emission: {} ms", EMIS_T.lock().unwrap().as_millis());
        println!("Transition: {} ms", TRAN_T.lock().unwrap().as_millis());
        println!("Collapse: {} ms", COLL_T.lock().unwrap().as_millis());
        println!("Combine: {} ms", COMB_T.lock().unwrap().as_millis());
        println!(
            "Combine Diploid: {} ms",
            COMBD_T.lock().unwrap().as_millis()
        );
        println!("",);
    }

    fn windows(&self, mut rng: impl Rng) -> Vec<((usize, usize), (Usize, Usize))> {
        let windows = windows_split::split(
            self.params.variants.view(),
            self.params.min_window_len_cm,
            &mut rng,
        );

        let overlap_len = self.cur_overlap_region_len;
        let mut hmm_windows = Vec::with_capacity(windows.len());

        #[cfg(feature = "obliv")]
        let mut prev_end_write_boundary = Usize::protect(0);

        #[cfg(not(feature = "obliv"))]
        let mut prev_end_write_boundary = 0;

        for (i, window) in windows.iter().enumerate() {
            let split_point = window.1;

            #[cfg(feature = "obliv")]
            let mut end_write_boundary = Usize::protect((split_point + overlap_len) as u64);

            #[cfg(not(feature = "obliv"))]
            let mut end_write_boundary = split_point + overlap_len;

            #[cfg(feature = "obliv")]
            let mut done = Bool::protect(false);

            for i in 0..overlap_len {
                if split_point + i >= self.genotype_graph.graph.len() {
                    break;
                }

                #[cfg(feature = "obliv")]
                {
                    let cond = self.genotype_graph.graph[split_point + i].is_segment_marker();
                    end_write_boundary = (cond & !done)
                        .select(Usize::protect((split_point + i) as u64), end_write_boundary);
                    done = cond.select(Bool::protect(true), done);
                }

                #[cfg(not(feature = "obliv"))]
                if self.genotype_graph.graph[split_point + i].is_segment_marker() {
                    end_write_boundary = split_point + i;
                    break;
                }
            }
            #[cfg(feature = "obliv")]
            let v = if i == 0 {
                (
                    (window.0, window.1 + overlap_len),
                    (Usize::protect(window.0 as u64), end_write_boundary),
                )
            } else if i == windows.len() - 1 {
                (
                    (window.0 - overlap_len, window.1),
                    (prev_end_write_boundary, Usize::protect(window.1 as u64)),
                )
            } else {
                (
                    (window.0 - overlap_len, window.1 + overlap_len),
                    (prev_end_write_boundary, end_write_boundary),
                )
            };

            #[cfg(not(feature = "obliv"))]
            let v = if i == 0 {
                (
                    (window.0, window.1 + overlap_len),
                    (window.0, end_write_boundary),
                )
            } else if i == windows.len() - 1 {
                (
                    (window.0 - overlap_len, window.1),
                    (prev_end_write_boundary, window.1),
                )
            } else {
                (
                    (window.0 - overlap_len, window.1 + overlap_len),
                    (prev_end_write_boundary, end_write_boundary),
                )
            };
            hmm_windows.push(v);
            prev_end_write_boundary = end_write_boundary;
        }
        hmm_windows
    }

    #[cfg(not(feature = "obliv"))]
    fn windows_full_segments(&self, mut rng: impl Rng) -> Vec<((usize, usize), (usize, usize))> {
        let windows = windows_split::split_by_segment(
            &self.genotype_graph,
            self.params.variants.view(),
            self.params.min_window_len_cm,
            &mut rng,
        );
        let mut hmm_windows = Vec::with_capacity(windows.len());
        let mut prev_end_write_boundary = 0;
        let mut start_boundary = 0;

        for (i, window) in windows.iter().enumerate() {
            let split_point = window.1;

            let mut end_write_boundary = None;
            let mut next_start_boundary = None;

            let mut j = 0;
            loop {
                if split_point + j >= self.genotype_graph.graph.len() {
                    break;
                }
                if self.genotype_graph.graph[split_point + j].is_segment_marker() {
                    end_write_boundary = Some(split_point + j);
                    break;
                }
                j += 1;
            }

            let mut j = 0;
            loop {
                if split_point < j + 1 {
                    break;
                }
                if self.genotype_graph.graph[split_point - j - 1].is_segment_marker() {
                    next_start_boundary = Some(split_point - j - 1);
                    break;
                }
                j += 1;
            }

            let v = if i == 0 {
                (
                    (window.0, end_write_boundary.unwrap()),
                    (window.0, end_write_boundary.unwrap()),
                )
            } else if i == windows.len() - 1 {
                (
                    (start_boundary, window.1),
                    (prev_end_write_boundary, window.1),
                )
            } else {
                (
                    (start_boundary, end_write_boundary.unwrap()),
                    (prev_end_write_boundary, end_write_boundary.unwrap()),
                )
            };

            if v.1 .0 != v.1 .1 {
                hmm_windows.push(v);
            }

            if i < windows.len() - 1 {
                prev_end_write_boundary = end_write_boundary.unwrap();
                start_boundary = next_start_boundary.unwrap();
            }
        }
        hmm_windows
    }

    fn get_ignored_sites(
        variants: ArrayView1<Variant>,
        genotypes: ArrayView1<Genotype>,
    ) -> Array1<Bool> {
        let mut ignored_sites = Array1::<Bool>::from_elem(variants.dim(), tp_value!(false, bool));
        Zip::from(&mut ignored_sites)
            .and(&variants)
            .and(&genotypes)
            .for_each(|s, v, &g| {
                let rarity = v.rarity();
                #[cfg(feature = "obliv")]
                {
                    *s = (g.tp_eq(&0) | g.tp_eq(&2))
                        & !((rarity == Rarity::NotRare)
                            | ((rarity == Rarity::Rare(true)) & g.tp_eq(&2))
                            | ((rarity == Rarity::Rare(false)) & g.tp_eq(&0)));
                }
                #[cfg(not(feature = "obliv"))]
                {
                    *s = (g == 0 || g == 2)
                        && !(rarity == Rarity::NotRare
                            || (rarity == Rarity::Rare(true) && g == 2)
                            || (rarity == Rarity::Rare(false) && g == 0));
                }
            });
        ignored_sites
    }
}

fn select_ref_panel(
    ref_panel: &RefPanelSlice,
    neighbors: &[Option<Vec<Usize>>],
) -> (Array2<Genotype>, usize) {
    #[cfg(feature = "obliv")]
    let neighbors_bitmap = {
        let mut bitmap = obliv_utils::bitmap::OblivBitmap::new(ref_panel.n_haps);
        bitmap.map_from_iter(
            neighbors
                .into_iter()
                .filter_map(|v| v.as_ref())
                .map(|v| v.iter().map(|&v| v.as_u32()))
                .flatten(),
        );
        bitmap
    };

    #[cfg(not(feature = "obliv"))]
    let neighbors_bitmap = {
        let mut bitmap = vec![false; ref_panel.n_haps];
        for &i in neighbors
            .into_iter()
            .filter_map(|v| v.as_ref())
            .map(|v| v.iter())
            .flatten()
        {
            bitmap[i] = true;
        }
        bitmap
    };

    #[cfg(feature = "obliv")]
    let k = neighbors_bitmap.iter().filter(|b| b.expose()).count();

    #[cfg(not(feature = "obliv"))]
    let k = neighbors_bitmap.iter().filter(|&&b| b).count();

    #[cfg(feature = "obliv")]
    return (
        ref_panel
            .filter(
                &neighbors_bitmap
                    .iter()
                    .map(|v| v.expose())
                    .collect::<Vec<_>>(),
            )
            .map(|&v| Genotype::protect(v)),
        k,
    );

    #[cfg(not(feature = "obliv"))]
    (ref_panel.filter(&neighbors_bitmap), k)
}

fn find_nn_bitmap(neighbors: &[Option<Vec<Usize>>], n_haps: usize) -> (Vec<Bool>, usize) {
    #[cfg(feature = "obliv")]
    let neighbors_bitmap = {
        let mut bitmap = obliv_utils::bitmap::OblivBitmap::new(n_haps);
        bitmap.map_from_iter(
            neighbors
                .into_iter()
                .filter_map(|v| v.as_ref())
                .map(|v| v.iter().map(|&v| v.as_u32()))
                .flatten(),
        );
        bitmap
    };

    #[cfg(not(feature = "obliv"))]
    let neighbors_bitmap = {
        let mut bitmap = vec![false; n_haps];
        for &i in neighbors
            .into_iter()
            .filter_map(|v| v.as_ref())
            .map(|v| v.iter())
            .flatten()
        {
            bitmap[i] = true;
        }
        bitmap
    };

    #[cfg(feature = "obliv")]
    let k = neighbors_bitmap.iter().filter(|b| b.expose()).count();

    #[cfg(not(feature = "obliv"))]
    let k = neighbors_bitmap.iter().filter(|&&b| b).count();

    #[cfg(feature = "obliv")]
    return (neighbors_bitmap.iter().collect(), k);

    #[cfg(not(feature = "obliv"))]
    (neighbors_bitmap, k)
}
