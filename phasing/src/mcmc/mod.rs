mod filter;
mod params;
mod sampling;
mod viterbi;
mod windows_split;
pub use params::*;

use crate::genotype_graph::GenotypeGraph;
use crate::hmm::combine_dips;
use crate::mcmc::filter::*;
use crate::memory::RealMemory;
use crate::variants::{Rarity, Variant};
use crate::{tp_value, Bool, Genotype, Real, Usize, U8};
use rand::Rng;

use std::time::{Duration, Instant};

use std::cell::RefCell;
thread_local! {
    pub static FILTER: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static HMM: RefCell<Duration> = RefCell::new(Duration::ZERO);
}

#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::{TpEq, TpI8, TpOrd};

use ndarray::{s, Array1, Array2, Array3, ArrayView1, Zip};

const N_HETS_PER_SEGMENT: usize = 3;
const P: usize = 1 << N_HETS_PER_SEGMENT;

#[derive(PartialEq, Eq, Clone, Debug)]
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
    prob_mask: Array3<Bool>,
    tprobs: Array3<Real>,
    #[cfg(feature = "obliv")]
    tprobs_e: Array2<TpI8>,
    #[cfg(not(feature = "obliv"))]
    n_old_segments: usize,
}

impl<'a> Mcmc<'a> {
    pub fn run(
        params: &'a McmcSharedParams,
        genotypes: ArrayView1<Genotype>,
        iterations: &[IterOption],
        mut rng: impl Rng,
        id: &str,
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

        let mut mcmc = Self::initialize(&params, genotypes.view(), id);

        let mut bprob_memory = RealMemory::new();

        for iter in iterations_iternal {
            mcmc.iteration(iter, &mut bprob_memory, &mut rng, id);
            println!("# adds = {}", tp_fixedpoint::NADD.with_borrow(|v| *v));
            println!("# subs = {}", tp_fixedpoint::NSUB.with_borrow(|v| *v));
            println!("# muls = {}", tp_fixedpoint::NMUL.with_borrow(|v| *v));
            println!("# divs = {}", tp_fixedpoint::NDIV.with_borrow(|v| *v));
        }

        #[cfg(feature = "obliv")]
        let epsilon = Real::protect_f32(1e-6);

        Zip::from(&mut mcmc.tprobs)
            .and(&mcmc.prob_mask)
            .for_each(|t, &m| {
                #[cfg(feature = "obliv")]
                {
                    *t = m.select(epsilon, *t);
                }

                #[cfg(not(feature = "obliv"))]
                if m {
                    *t = 1e-6;
                }
            });

        mcmc.phased_ind = viterbi::viterbi(mcmc.tprobs.view(), mcmc.genotype_graph.graph.view());

        mcmc.genotype_graph
            .traverse_graph_pair(mcmc.phased_ind.view(), mcmc.estimated_haps.view_mut());
        println!("# adds = {}", tp_fixedpoint::NADD.with_borrow(|v| *v));
        println!("# subs = {}", tp_fixedpoint::NSUB.with_borrow(|v| *v));
        println!("# muls = {}", tp_fixedpoint::NMUL.with_borrow(|v| *v));
        println!("# divs = {}", tp_fixedpoint::NDIV.with_borrow(|v| *v));

        mcmc.estimated_haps
    }

    fn initialize(params: &'a McmcSharedParams, genotypes: ArrayView1<Genotype>, id: &str) -> Self {
        println!("=== Initialization ({id}) ===",);
        let now = Instant::now();

        let (h_0, h_1) = crate::neighbor_finding::mcmc_init(
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

        let prob_mask = Array3::from_elem(tprobs.dim(), tp_value!(false, bool));

        #[cfg(feature = "obliv")]
        let tprobs_e = Array2::<TpI8>::from_elem((estimated_haps.nrows(), P), TpI8::protect(0));

        let genotype_graph = GenotypeGraph::build(genotypes);

        #[cfg(not(feature = "obliv"))]
        let n_old_segments = genotype_graph.n_segments();

        let ignored_sites = Self::get_ignored_sites(params.variants.view(), genotypes);
        println!("Initialization: {} ms", (Instant::now() - now).as_millis());
        println!("",);
        Self {
            params,
            cur_overlap_region_len: params.overlap_region_len,
            genotype_graph,
            estimated_haps,
            phased_ind,
            prob_mask,
            tprobs,
            ignored_sites,
            #[cfg(feature = "obliv")]
            tprobs_e,
            #[cfg(not(feature = "obliv"))]
            n_old_segments,
        }
    }

    fn iteration(
        &mut self,
        iter_option: IterOptionInternal,
        bprob_memory: &mut RealMemory,
        mut rng: impl Rng,
        id: &str,
    ) {
        println!("=== {:?} Iteration ({id}) ===", iter_option);
        let now = Instant::now();

        let pbwt_group_filter = self.params.randomize_pbwt_group_bitmask(&mut rng);

        let neighbors = {
            let pbwt_group_filter = pbwt_group_filter
                .into_iter()
                .zip(&self.params.pbwt_evaluted)
                .map(|(a, &b)| a && b)
                .collect::<Vec<_>>();

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

            let mut nn_0 = crate::neighbor_finding::find_top_neighbors(
                &h_0,
                self.params.s,
                &self.params.pbwt_tries,
                self.params.ref_panel.n_haps,
                &pbwt_group_filter,
            );

            let nn_1 = crate::neighbor_finding::find_top_neighbors(
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

        {
            println!("Neighbors Finding: {} ms", now.elapsed().as_millis());

            #[cfg(feature = "benchmarking")]
            {
                println!(
                    "\tInsert target: {:?} ms",
                    crate::neighbor_finding::timing::INSERT
                        .with(|v| {
                            let out = *v.borrow();
                            *v.borrow_mut() = std::time::Duration::ZERO;
                            out
                        })
                        .as_millis()
                );
                println!(
                    "\tInitialize ranks: {:?} ms",
                    crate::neighbor_finding::timing::INIT_RANKS
                        .with(|v| {
                            let out = *v.borrow();
                            *v.borrow_mut() = std::time::Duration::ZERO;
                            out
                        })
                        .as_millis()
                );
                println!(
                    "\tNN merging & lookups: {:?} ms",
                    crate::neighbor_finding::timing::LOOKUP
                        .with(|v| {
                            let out = *v.borrow();
                            *v.borrow_mut() = std::time::Duration::ZERO;
                            out
                        })
                        .as_millis()
                );
                println!(
                    "\tUpdate PBWT between blocks: {:?} ms",
                    crate::neighbor_finding::timing::UPDATE
                        .with(|v| {
                            let out = *v.borrow();
                            *v.borrow_mut() = std::time::Duration::ZERO;
                            out
                        })
                        .as_millis()
                );
            }
        }

        let windows = self.windows(&mut rng);

        #[cfg(feature = "obliv")]
        let mut prev_ind = (U8::protect(0), U8::protect(0));
        #[cfg(not(feature = "obliv"))]
        let mut prev_ind = (0, 0);
        let mut sum_window_size = 0;
        let n_windows = windows.len();
        let mut ks = Vec::new();
        let mut max_ks = Vec::new();
        let mut window_sizes = Vec::new();
        let mut tprob_pairs = Array2::<Real>::zeros((P, P));

        let mut i = 0;

        let rprobs = self.params.hmm_params.get_rprobs(
            self.ignored_sites.view(),
            &self.genotype_graph,
            self.params.variants.view(),
        );

        for (window_i, ((start_w, end_w), (start_write_w, end_write_w))) in
            windows.into_iter().enumerate()
        {
            sum_window_size +=
                self.params.variants[end_w - 1].bp - self.params.variants[start_w].bp;
            let genotype_graph_w = self.genotype_graph.slice(start_w, end_w);
            let params_w = self.params.slice(start_w, end_w);
            let rprobs_w = rprobs.slice(start_w, end_w);

            let neighbors_w = &neighbors[start_w..end_w];

            let max_k = neighbors_w
                .iter()
                .filter_map(|v| v.as_ref().map(|v| v.len()))
                .sum::<usize>();
            max_ks.push(max_k as f64);

            window_sizes.push((end_w - start_w) as f64);

            let t = Instant::now();
            let (unfolded, filter, n_full_states) =
                filter_blocks(neighbors_w, &params_w.ref_panel.blocks);

            //TODO remove this
            ks.push(n_full_states.expose() as f64);

            FILTER.with(|v| {
                let mut v = v.borrow_mut();
                *v += t.elapsed();
            });

            #[cfg(feature = "obliv")]
            let (tprobs_window, tprobs_window_e) = crate::hmm::Hmm::forward_backward(
                unfolded.view(),
                filter.view(),
                n_full_states,
                genotype_graph_w.graph.view(),
                &self.params.hmm_params,
                &rprobs_w,
                Array1::from_elem(params_w.ref_panel.n_sites, tp_value!(true, bool)).view(),
                start_w == 0,
                bprob_memory,
            );

            #[cfg(not(feature = "obliv"))]
            let tprobs_window = crate::hmm::Hmm::forward_backward(
                unfolded.view(),
                filter.view(),
                n_full_states,
                genotype_graph_w.graph.view(),
                &self.params.hmm_params,
                &rprobs_w,
                Array1::from_elem(params_w.ref_panel.n_sites, tp_value!(true, bool)).view(),
                start_w == 0,
            );

            #[cfg(feature = "obliv")]
            let tprobs_window_src = tprobs_window.view();

            #[cfg(feature = "obliv")]
            let tprobs_window_src_e = tprobs_window_e.view();

            #[cfg(feature = "obliv")]
            let mut tprobs_window_target = self.tprobs.slice_mut(s![start_w..end_w, .., ..]);

            #[cfg(feature = "obliv")]
            let mut prob_mask_target = self.prob_mask.slice_mut(s![start_w..end_w, .., ..]);

            #[cfg(not(feature = "obliv"))]
            let tprobs_window_src =
                tprobs_window.slice_mut(s![start_write_w - start_w..end_write_w - start_w, .., ..]);

            #[cfg(not(feature = "obliv"))]
            let genotype_graph = &self.genotype_graph;

            #[cfg(not(feature = "obliv"))]
            let mut tprobs_window_target =
                self.tprobs
                    .slice_mut(s![start_write_w..end_write_w, .., ..]);

            #[cfg(not(feature = "obliv"))]
            let mut prob_mask_target =
                self.prob_mask
                    .slice_mut(s![start_write_w..end_write_w, .., ..]);

            #[cfg(feature = "obliv")]
            let mut j = 0u64;

            #[cfg(feature = "obliv")]
            let epsilon = Real::protect_f32(1e-6);

            if iter_option == IterOptionInternal::Pruning {
                Zip::from(tprobs_window_target.outer_iter_mut())
                    .and(tprobs_window_src.outer_iter())
                    .for_each(|mut t, s| {
                        #[cfg(feature = "obliv")]
                        {
                            let cond = (start_write_w - start_w as u64).tp_lt_eq(&j)
                                & (end_write_w - start_w as u64).tp_gt(&j);
                            let s_e = tprobs_window_src_e.slice(s![j as usize, .., ..]);
                            combine_dips(s, s_e, t.view_mut(), cond);
                            j += 1;
                        }

                        #[cfg(not(feature = "obliv"))]
                        if i == 0 || genotype_graph.graph[i].is_segment_marker() {
                            combine_dips(s, t.view_mut());
                        } else {
                            let mut t = t;
                            t.fill(0.);
                        }
                        i += 1;
                    });
            } else if iter_option == IterOptionInternal::Main(true) {
                Zip::from(tprobs_window_target.outer_iter_mut())
                    .and(tprobs_window_src.outer_iter())
                    .and(prob_mask_target.outer_iter_mut())
                    .for_each(|mut t, s, mut m| {
                        #[cfg(feature = "obliv")]
                        {
                            let cond = (start_write_w - start_w as u64).tp_lt_eq(&j)
                                & (end_write_w - start_w as u64).tp_gt(&j);
                            let s_e = tprobs_window_src_e.slice(s![j as usize, .., ..]);
                            combine_dips(s, s_e, t.view_mut(), cond);
                            Zip::from(&t)
                                .and(&mut m)
                                .for_each(|&t, m| *m = t.tp_lt(&epsilon) & cond);
                            j += 1;
                        }

                        #[cfg(not(feature = "obliv"))]
                        if i == 0 || genotype_graph.graph[i].is_segment_marker() {
                            combine_dips(s, t.view_mut());
                            Zip::from(&t).and(&mut m).for_each(|&t, m| *m = t <= 1e-6);
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
                tprobs_window_src_e.view(),
                genotype_graph_w.graph.view(),
                window_i == 0,
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

            HMM.with(|v| {
                let mut v = v.borrow_mut();
                *v += t.elapsed();
            });
        }

        self.genotype_graph
            .traverse_graph_pair(self.phased_ind.view(), self.estimated_haps.view_mut());

        if iter_option == IterOptionInternal::Pruning {
            self.genotype_graph.prune(self.tprobs.view());

            #[cfg(not(feature = "obliv"))]
            {
                let n_new_segments = self.genotype_graph.n_segments();
                println!("#new-segments: {n_new_segments}",);
                println!(
                    "Trimming: {:.2}%",
                    (1. - n_new_segments as f64 / self.n_old_segments as f64) * 100.
                );
            }
        }

        use statrs::statistics::Statistics;
        println!(
            "K: {:.3}+/-{:.3}",
            Statistics::mean(&ks),
            Statistics::std_dev(&ks)
        );
        println!(
            "Max K: {:.3}+/-{:.3}",
            Statistics::mean(&max_ks),
            Statistics::std_dev(&max_ks)
        );
        println!(
            "Window sizes (#sites): {:.3}+/-{:.3}",
            Statistics::mean(&window_sizes),
            Statistics::std_dev(&window_sizes)
        );

        println!(
            "Total window size (Mb): {:.2} Mb",
            sum_window_size as f64 / n_windows as f64 / 1e6
        );

        println!(
            "HMM+Filter: {:?} ms",
            HMM.with(|v| {
                let out = *v.borrow();
                *v.borrow_mut() = std::time::Duration::ZERO;
                out
            })
            .as_millis()
        );

        #[cfg(feature = "benchmarking")]
        println!(
            "\tFilter Ref Panel: {:?} ms",
            FILTER
                .with(|v| {
                    let out = *v.borrow();
                    *v.borrow_mut() = std::time::Duration::ZERO;
                    out
                })
                .as_millis()
        );

        #[cfg(feature = "benchmarking")]
        {
            println!(
                "\tEmission: {:?} ms",
                crate::hmm::EMISS
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "\tTransition: {:?} ms",
                crate::hmm::TRANS
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "\tCollapse: {:?} ms",
                crate::hmm::COLL
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "\tCombine: {:?} ms",
                crate::hmm::COMB
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );
            println!(
                "\tMemory allocation: {:?} ms",
                crate::hmm::MEMORY
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );
        }

        println!("Elapsed: {} ms", now.elapsed().as_millis());
        println!();
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
                    (
                        window.0,
                        (window.1 + overlap_len).min(self.params.variants.len()),
                    ),
                    (Usize::protect(window.0 as u64), end_write_boundary),
                )
            } else if i == windows.len() - 1 {
                (
                    (window.0.saturating_sub(overlap_len), window.1),
                    (prev_end_write_boundary, Usize::protect(window.1 as u64)),
                )
            } else {
                (
                    (
                        window.0.saturating_sub(overlap_len),
                        (window.1 + overlap_len).min(self.params.variants.len()),
                    ),
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
