mod params;
mod sampling;
mod viterbi;
mod windows_split;
pub use params::*;

use crate::genotype_graph::GenotypeGraph;
use crate::hmm::combine_dips;
use crate::variants::{Rarity, Variant};
use crate::{tp_value, Bool, Genotype, Real, UInt, Usize, U8};
//use common::ref_panel::RefPanelSlice;
use rand::Rng;

use std::time::{Duration, Instant};

use std::cell::RefCell;
thread_local! {
    pub static FILTER: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static FULLFWBW: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static REDFWBW: RefCell<Duration> = RefCell::new(Duration::ZERO);
}

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

        for iter in iterations_iternal {
            mcmc.iteration(iter, &mut rng, id);

            println!(
                "Emission: {:?} ms",
                crate::rss_hmm::reduced_obliv::EMISS
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "Transition: {:?} ms",
                crate::rss_hmm::reduced_obliv::TRANS
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "Collapse: {:?} ms",
                crate::rss_hmm::reduced_obliv::COLL
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "Combine 1: {:?} ms",
                crate::rss_hmm::reduced_obliv::COMB1
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "Combine 2: {:?} ms",
                crate::rss_hmm::reduced_obliv::COMB2
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "Expand: {:?} ms",
                crate::rss_hmm::reduced_obliv::EXPAND
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "Block Transition: {:?} ms",
                crate::rss_hmm::reduced_obliv::BLOCK
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "Filter Ref Panel: {:?} ms",
                FILTER
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "Full HMM: {:?} ms",
                FULLFWBW
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );

            println!(
                "RSS HMM: {:?} ms",
                REDFWBW
                    .with(|v| {
                        let out = *v.borrow();
                        *v.borrow_mut() = std::time::Duration::ZERO;
                        out
                    })
                    .as_millis()
            );
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

        mcmc.estimated_haps
    }

    fn initialize(params: &'a McmcSharedParams, genotypes: ArrayView1<Genotype>, id: &str) -> Self {
        println!("=== Initialization ({id}) ===",);
        let now = Instant::now();
        #[cfg(feature = "obliv")]
        use compressed_pbwt_obliv::mcmc_init::mcmc_init;

        //let (h_0, h_1) = mcmc_init_loaded("init.txt");

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

        let prob_mask = Array3::from_elem(tprobs.dim(), tp_value!(false, bool));

        #[cfg(feature = "obliv")]
        let tprobs_e = Array2::<TpI8>::from_elem((estimated_haps.nrows(), P), TpI8::protect(0));

        let genotype_graph = GenotypeGraph::build(genotypes);

        //{
        //use std::io::Write;
        //crate::DEBUG_FILE.with(|f| {
        //let mut f = f.borrow_mut();
        //writeln!(*f, "## genotype graph ##").unwrap();
        //let mut count = 0;
        //for g in &genotype_graph.graph {
        //if g.is_segment_marker() {
        //writeln!(*f, "{count}").unwrap();
        //count = 0;
        //}
        //count += 1;
        //}
        //writeln!(*f, "{count}").unwrap();
        //});
        //}

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

    fn iteration(&mut self, iter_option: IterOptionInternal, mut rng: impl Rng, id: &str) {
        println!("=== {:?} Iteration ({id}) ===", iter_option);
        let now = Instant::now();

        let pbwt_group_filter = self.params.randomize_pbwt_group_bitmask(&mut rng);
        //let pbwt_group_filter = {
        //use std::io::BufRead;
        //let f = std::io::BufReader::new(std::fs::File::open("random_pbwt.txt").unwrap());
        //f.lines()
        //.map(|l| l.unwrap().parse::<u8>().unwrap() == 1)
        //.collect::<Vec<_>>()
        //};

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

            #[cfg(feature = "obliv")]
            use compressed_pbwt_obliv::nn::find_top_neighbors;

            #[cfg(not(feature = "obliv"))]
            use compressed_pbwt::nn::find_top_neighbors;
            //use crate::neighbors_finding::find_top_neighbors;

            let mut nn_0 = find_top_neighbors(
                &h_0,
                self.params.s,
                &self.params.pbwt_tries,
                //self.params.ref_panel.iter(),
                self.params.ref_panel.n_haps,
                &pbwt_group_filter,
            );

            //nn_0[3] = Some(vec![6054 - 2, 527 - 2, 19938 - 2, 19854 - 2, 19773 - 2]);

            let nn_1 = find_top_neighbors(
                &h_1,
                self.params.s,
                &self.params.pbwt_tries,
                //self.params.ref_panel.iter(),
                self.params.ref_panel.n_haps,
                &pbwt_group_filter,
            );
            //{
            //use std::io::Write;
            //crate::DEBUG_FILE.with(|f| {
            //let mut f = f.borrow_mut();
            //writeln!(*f, "## neighbors ##").unwrap();

            //for (i, ((a, &h_0), (b, &h_1))) in nn_0
            //.iter()
            //.zip(h_0.iter())
            //.zip(nn_1.iter().zip(h_1.iter()))
            //.enumerate()
            //{
            //if let Some(a) = a.as_ref() {
            //write!(*f, "{i}, 0, {}: ", h_0 as u8).unwrap();
            //for v in a {
            //let v = v + 2;
            //write!(*f, "{v} ").unwrap();
            //}
            //writeln!(*f).unwrap();
            //}
            //if let Some(b) = b.as_ref() {
            //write!(*f, "{i}, 1, {}: ", h_1 as u8).unwrap();
            //for v in b {
            //let v = v + 2;
            //write!(*f, "{v} ").unwrap();
            //}
            //writeln!(*f).unwrap();
            //}
            //}
            //});
            //}

            for (a, b) in nn_0.iter_mut().zip(nn_1.into_iter()) {
                a.as_mut().map(|v| v.extend(b.unwrap().into_iter()));
            }

            nn_0
        };
        println!(
            "Neighbors Finding: {} ms",
            (Instant::now() - now).as_millis()
        );

        let windows = self.windows(&mut rng);

        //let windows = {
        //use std::io::BufRead;
        //let f = std::io::BufReader::new(std::fs::File::open("windows.txt").unwrap());
        //let mut prev_end = 0;
        //f.lines()
        //.map(|l| {
        //let l = l.unwrap();
        //let mut l = l.split_whitespace();
        //let start_w = l.next().unwrap().parse::<usize>().unwrap();
        //let end_w = l.next().unwrap().parse::<usize>().unwrap() + 1;

        //let start_write_w = if prev_end != 0 { prev_end } else { start_w };

        //prev_end = end_w;

        //((start_w, end_w), (start_write_w, end_w))
        //})
        //.collect::<Vec<_>>()
        //};

        #[cfg(feature = "obliv")]
        let mut prev_ind = (U8::protect(0), U8::protect(0));
        #[cfg(not(feature = "obliv"))]
        let mut prev_ind = (0, 0);
        let mut sum_window_size = 0;
        let n_windows = windows.len();
        let mut ks = Vec::new();
        let mut max_ks = Vec::new();
        let mut tprob_pairs = Array2::<Real>::zeros((P, P));

        let mut i = 0;
        let mut is_first_window = true;

        let rprobs = self.params.hmm_params.get_rprobs(
            self.ignored_sites.view(),
            &self.genotype_graph,
            self.params.variants.view(),
        );
        //let mut windows_count = 0;

        for ((start_w, end_w), (start_write_w, end_write_w)) in windows.into_iter() {
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

            //{
            //let mut n_set = std::collections::HashSet::new();
            //for n in neighbors_w {
            //if let Some(n) = n.as_ref() {
            //for j in n {
            //n_set.insert(j);
            //}
            //}
            //}
            //let mut n_list = n_set.into_iter().collect::<Vec<_>>();
            //n_list.sort();

            //use std::io::Write;
            //crate::DEBUG_FILE.with(|f| {
            //let mut f = f.borrow_mut();
            //if windows_count == 0 {
            //writeln!(*f, "## kstates ##").unwrap();
            //}
            //writeln!(*f, "window {windows_count}").unwrap();
            //for j in n_list {
            //writeln!(*f, "{}", j + 2).unwrap();
            //}
            //});

            //windows_count += 1;
            //}

            let (full_filter, k) = find_nn_bitmap(neighbors_w, params_w.ref_panel.n_haps);

            #[cfg(feature = "obliv")]
            {
                ks.push(k.expose() as f64);
            }
            #[cfg(not(feature = "obliv"))]
            {
                ks.push(k as f64);
            }

            //{
            //let t = Instant::now();
            //let (max_k_neighbors, _filter) = neighbors_to_filter(&neighbors);
            //let mut unfolded = Array2::from_elem(
            //(end_w - start_w, max_k_neighbors.len()),
            //Genotype::protect(0),
            //);

            //let mut start_slice = 0;
            //for block in &params_w.ref_panel.blocks {
            //unfold_block(
            //block,
            //&max_k_neighbors,
            //unfolded.slice_mut(s![start_slice..start_slice + block.n_sites(), ..]),
            //);
            //start_slice += block.n_sites();
            //}
            //FILTER.with(|v| {
            //let mut v = v.borrow_mut();
            //*v += t.elapsed();
            //});

            //let t = Instant::now();
            //let mut hmm = crate::hmm::Hmm::new();
            //hmm.forward_backward(
            //unfolded.view(),
            //genotype_graph_w.graph.view(),
            //&self.params.hmm_params,
            //&rprobs_w,
            //Array1::from_elem(params_w.ref_panel.n_sites, tp_value!(false, bool)).view(),
            //start_w == 0,
            //);
            //FULLFWBW.with(|v| {
            //let mut v = v.borrow_mut();
            //*v += t.elapsed();
            //});
            //}

            let t = Instant::now();
            let filtered_blocks = params_w
                .ref_panel
                .blocks
                .iter()
                .map(|b| {
                    crate::rss_hmm::filtered_block::FilteredBlockSliceObliv::from_block_slice(
                        b,
                        &full_filter,
                    )
                })
                .collect::<Vec<_>>();

            let fwbw_out = crate::rss_hmm::reduced_obliv::HmmReduced::fwbw(
                &filtered_blocks,
                params_w.ref_panel.n_sites,
                k,
                params_w.ref_panel.n_haps,
                genotype_graph_w.graph.view(),
                self.params.hmm_params.eprob,
                &rprobs_w,
                Array1::from_elem(params_w.ref_panel.n_sites, tp_value!(false, bool)).view(),
            );
            REDFWBW.with(|v| {
                let mut v = v.borrow_mut();
                *v += t.elapsed();
            });

            //let filtered_blocks = params_w
            //.ref_panel
            //.blocks
            //.iter()
            //.map(|b| {
            //crate::rss_hmm::filtered_block::FilteredBlockSlice::from_block_slice(
            //b,
            //&full_filter,
            //)
            //})
            //.collect::<Vec<_>>();

            //let fwbw_out = crate::rss_hmm::HmmReduced::fwbw(
            //&filtered_blocks,
            //params_w.ref_panel.n_sites,
            //genotype_graph_w.graph.view(),
            //self.params.hmm_params.eprob,
            //&rprobs_w,
            //Array1::from_elem(params_w.ref_panel.n_sites, tp_value!(false, bool)).view(),
            //);

            #[cfg(feature = "obliv")]
            let (first_tprobs, first_tprobs_e, tprobs, tprobs_e) = fwbw_out;

            #[cfg(not(feature = "obliv"))]
            let (first_tprobs, tprobs) = fwbw_out;

            let mut tprobs_window = tprobs;
            #[cfg(feature = "obliv")]
            let mut tprobs_window_e = tprobs_e;

            if is_first_window {
                Zip::from(tprobs_window.slice_mut(s![0, .., ..]).rows_mut())
                    .for_each(|mut r| r.assign(&first_tprobs));

                #[cfg(feature = "obliv")]
                Zip::from(tprobs_window_e.slice_mut(s![0, .., ..]).rows_mut())
                    .for_each(|mut r| r.assign(&first_tprobs_e));
            }

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

        //{
        //use std::io::Write;
        //crate::DEBUG_FILE.with(|f| {
        //let mut f = f.borrow_mut();
        //writeln!(*f, "## transitions ##").unwrap();
        ////println!("# transisions = {}", self.genotype_graph.graph.iter().filter(|v| v.is_segment_marker()).count());
        ////write!(*f, "0: ").unwrap();
        ////for i in &first_tprobs {
        ////write!(*f, "{:.8e} ", i).unwrap();
        ////}
        ////writeln!(*f).unwrap();
        //for (c, t) in self
        //.tprobs
        //.outer_iter()
        //.into_iter()
        //.zip(self.genotype_graph.graph.iter())
        //.filter_map(|(t, g)| if g.is_segment_marker() { Some(t) } else { None })
        //.enumerate()
        //{

        //write!(*f, "{}: ", c+1).unwrap();
        //for i in &t {
        //write!(*f, "{:.3e} ", i).unwrap();
        //}
        //writeln!(*f).unwrap();
        //}
        //});
        //}

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

            self.cur_overlap_region_len *= 2;
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
            "Window size: {:.2} Mb",
            sum_window_size as f64 / n_windows as f64 / 1e6
        );
        println!("Elapsed: {} ms", (Instant::now() - now).as_millis());
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

fn find_nn_bitmap(neighbors: &[Option<Vec<Usize>>], n_haps: usize) -> (Vec<Bool>, UInt) {
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
    let k = neighbors_bitmap
        .iter()
        .fold(UInt::protect(0), |acc, v| acc + v.as_u32());

    #[cfg(not(feature = "obliv"))]
    let k = neighbors_bitmap.iter().filter(|&&b| b).count() as u32;

    #[cfg(feature = "obliv")]
    return (neighbors_bitmap.iter().collect(), k);

    #[cfg(not(feature = "obliv"))]
    (neighbors_bitmap, k)
}

//fn mcmc_init_loaded(path: &str) -> (Vec<Bool>, Vec<Bool>) {
//use std::io::BufRead;
//let f = std::io::BufReader::new(std::fs::File::open(std::path::Path::new(path)).unwrap());
//let mut h0 = Vec::new();
//let mut h1 = Vec::new();
//for line in f.lines() {
//let line = line.unwrap();
//let mut line = line.split_whitespace();
//let _h0 = line.next().unwrap().parse::<u8>().unwrap();
//let _h1 = line.next().unwrap().parse::<u8>().unwrap();
//h0.push(_h0 == 1);
//h1.push(_h1 == 1);
//}
//(h0, h1)
//}

fn neighbors_to_filter(neighbors: &[Option<Vec<Usize>>]) -> (Vec<Usize>, Vec<Bool>) {
    let mut neighbors = neighbors
        .into_iter()
        .filter_map(|v| v.as_ref())
        .flatten()
        .cloned()
        .collect::<Vec<Usize>>();
    obliv_utils::bitonic_sort::bitonic_sort(&mut neighbors, true);

    let filter = {
        let mut filter = vec![Bool::protect(false); neighbors.len()];
        let mut prev = neighbors[0];
        filter[0] = Bool::protect(true);

        for (f, n) in filter.iter_mut().zip(neighbors.iter()).skip(1) {
            let cond = prev.tp_not_eq(n);
            *f = cond;
            prev = *n;
        }
        filter
    };
    (neighbors, filter)
}

fn unfold_block<'a>(
    block: &common::ref_panel::BlockSlice<'a>,
    neighbors: &[Usize],
    mut unfolded: ndarray::ArrayViewMut2<Genotype>,
) {
    assert_eq!(unfolded.nrows(), block.n_sites());
    assert_eq!(unfolded.ncols(), neighbors.len());

    use std::iter::FromIterator;
    use tp_fixedpoint::timing_shield::TpU16;

    let obliv_index_map =
        obliv_utils::vec::OblivVec::from_iter(block.index_map.iter().map(|&v| TpU16::protect(v)));

    let unique_neighbors = neighbors
        .into_iter()
        .map(|&v| obliv_index_map.get(v.as_u32()))
        .collect::<Vec<_>>();
    Zip::from(block.haplotypes.rows())
        .and(unfolded.rows_mut())
        .for_each(|h, u| {
            unfold_haps(h, block.n_unique(), &unique_neighbors[..], u);
        })
}

fn unfold_haps(
    haps: ArrayView1<u8>,
    n_unique_haps: usize,
    unique_neighbors: &[tp_fixedpoint::timing_shield::TpU16],
    mut unfolded: ndarray::ArrayViewMut1<Genotype>,
) {
    use tp_fixedpoint::timing_shield::TpU64;

    let mut inner = obliv_utils::vec::OblivVec::with_capacity(haps.len().div_ceil(8));
    let haps = haps.as_slice().unwrap();
    for chunk in haps.chunks(8) {
        let mut new_u64 = 0u64;
        for &i in chunk {
            new_u64 <<= 8;
            new_u64 |= i as u64;
        }
        inner.push(TpU64::protect(new_u64));
    }

    let bitmap = obliv_utils::bitmap::OblivBitmap::from_inner(inner, n_unique_haps);

    unfolded
        .iter_mut()
        .zip(unique_neighbors.into_iter())
        .for_each(|(b, n)| {
            *b = bitmap.get(n.as_u32()).as_i8();
        });
}
