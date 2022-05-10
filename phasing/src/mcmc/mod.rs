mod initialize;
mod params;
pub mod sampling;
mod viterbi;
mod windows_split;
pub use params::*;

use crate::genotype_graph::GenotypeGraph;
use crate::hmm::{Hmm, HmmNew};
use crate::hmm::{COLL_T, COMBD_T, COMB_T, EMIS_T, TRAN_T};
use crate::neighbors_finding;
use crate::neighbors_finding::{NEIGHBOR_T, PBWT_T};
use crate::pbwt::EXPAND_T;
use crate::ref_panel::RefPanelSlice;
use crate::variants::{Rarity, Variant};
use crate::{tp_value_new, BoolMcc, Genotype, Real};
use rand::Rng;

use std::time::{Duration, Instant};

#[cfg(feature = "leak-resist-new")]
use tp_fixedpoint::timing_shield::TpI8;

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Zip};

const N_HETS_PER_SEGMENT: usize = 3;
const P: usize = 1 << N_HETS_PER_SEGMENT;

#[cfg(feature = "leak-resist-new")]
const MAX_K_OVERFLOW: usize = 1 << (63 - crate::F);

const MAX_K: usize = 2048;

#[derive(PartialEq, Eq, Clone)]
pub enum IterOption {
    Burnin(usize),
    Pruning(usize),
    Main(usize),
}

#[derive(PartialEq, Eq, Debug, Clone)]
enum IterOptionInternal {
    Burnin,
    Pruning(bool), // First pruning iteration?
    Main(bool),    // First main iteration?
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum IterOptionInternalNew {
    Burnin,
    Pruning(bool),    // first?
    Main(bool, bool), // (first, last)?
}

impl IterOptionInternalNew {
    pub fn is_sample(self) -> bool {
        match self {
            Self::Burnin | Self::Pruning(_) | Self::Main(_, false) => true,
            _ => false,
        }
    }

    pub fn is_save(self) -> bool {
        match self {
            Self::Pruning(_) | Self::Main(_, _) => true,
            _ => false,
        }
    }

    pub fn is_first_save(self) -> bool {
        match self {
            Self::Pruning(true) | Self::Main(true, _) => true,
            _ => false,
        }
    }
}

pub struct Mcmc<'a> {
    params: &'a McmcSharedParams,
    cur_overlap_region_len: usize,
    genotype_graph: GenotypeGraph,
    ignored_sites: Array1<BoolMcc>,
    estimated_haps: Array2<Genotype>,
    first_ind: u8,
    phased_ind: Array1<u8>,
    tprobs: Array3<Real>,
    #[cfg(feature = "leak-resist-new")]
    tprobs_e: Array2<TpI8>,
    first_tprobs_dip: Array1<f64>,
    tprobs_dip: Array3<f64>,
}

impl<'a> Mcmc<'a> {
    pub fn run_new(
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
                        iterations_iternal.push(IterOptionInternalNew::Burnin);
                    }
                }
                IterOption::Pruning(r) => {
                    let r = *r;
                    assert!(r > 0);
                    iterations_iternal.push(IterOptionInternalNew::Pruning(true));
                    for _ in 1..r {
                        iterations_iternal.push(IterOptionInternalNew::Pruning(false));
                    }
                }
                IterOption::Main(r) => {
                    let r = *r;
                    assert!(r > 0);
                    if r == 1 {
                        iterations_iternal.push(IterOptionInternalNew::Main(true, true));
                    } else {
                        iterations_iternal.push(IterOptionInternalNew::Main(true, false));
                        for _ in 1..(r - 1) {
                            iterations_iternal.push(IterOptionInternalNew::Main(false, false));
                        }
                        iterations_iternal.push(IterOptionInternalNew::Main(false, true));
                    }
                }
            }
        }

        //let mut ref_file = std::io::BufReader::new(
        //std::fs::File::open(
        //"/home/ndokmai/workspace/phasing-oram/phasing/tests/ref_tprobs_chr1.bin",
        //)
        //.unwrap(),
        //);

        //let mut ref_file = std::io::BufWriter::new(
        //std::fs::File::create(
        //"/home/ndokmai/workspace/phasing-oram/phasing/tests/ref_tprobs_chr1.bin",
        //)
        //.unwrap(),
        //);

        let mut mcmc = Self::initialize_new(&params, genotypes.view());

        //mcmc.save_estimated_haps(&mut ref_file);
        //mcmc.check_estimated_haps(&mut ref_file);

        for i in iterations_iternal {
            mcmc.iteration_new(i, &mut rng);
            //mcmc.save_tprobs(&mut ref_file);
            //mcmc.save_estimated_haps(&mut ref_file);
            //mcmc.check_tprobs(&mut ref_file);
            //mcmc.check_estimated_haps(&mut ref_file);
            //println!("{:#?}", mcmc.tprobs.slice(s![..10, .., ..]));
            //unimplemented!();
        }

        mcmc.phased_ind = viterbi::viterbi_new(
            mcmc.first_tprobs_dip.view(),
            mcmc.tprobs_dip.view(),
            mcmc.genotype_graph.graph.view(),
        );

        let first_ind = mcmc.phased_ind[0];
        let phased_ind = mcmc.phased_ind.slice(s![1..]).to_owned();

        mcmc.genotype_graph.traverse_graph_dip(
            first_ind,
            phased_ind.view(),
            mcmc.estimated_haps.view_mut(),
        );

        //mcmc.save_estimated_haps(&mut ref_file);
        //mcmc.check_estimated_haps(&mut ref_file);

        mcmc.estimated_haps
    }
    pub fn test_new(
        params: &'a McmcSharedParams,
        genotypes: ArrayView1<Genotype>,
        mut rng: impl Rng,
    ) -> (u8, Array1<u8>, Array2<i8>) {
        let mut mcmc = Self::initialize_new(&params, genotypes.view());

        mcmc.iteration_new(IterOptionInternalNew::Burnin, &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Burnin, &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Burnin, &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Burnin, &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Burnin, &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Burnin, &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Pruning(true), &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Burnin, &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Pruning(true), &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Burnin, &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Pruning(true), &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Main(true, false), &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Main(false, false), &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Main(false, false), &mut rng);
        mcmc.iteration_new(IterOptionInternalNew::Main(false, false), &mut rng);
        (mcmc.first_ind, mcmc.phased_ind, mcmc.estimated_haps)
    }

    pub fn test(
        params: &'a McmcSharedParams,
        genotypes: ArrayView1<Genotype>,
        mut rng: impl Rng,
    ) -> (u8, Array1<u8>, Array2<i8>) {
        let mut mcmc = Self::initialize(&params, genotypes.view());
        mcmc.iteration(IterOptionInternal::Burnin, &mut rng);
        let first_ind = mcmc.phased_ind[0];

        let phased_ind = Array1::from_vec(
            mcmc.phased_ind
                .iter()
                .zip(mcmc.genotype_graph.graph.iter())
                .filter_map(|(&i, g)| if g.is_segment_marker() { Some(i) } else { None })
                .collect(),
        );

        (first_ind, phased_ind, mcmc.estimated_haps)
    }

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
                    iterations_iternal.push(IterOptionInternal::Pruning(true));
                    for _ in 1..*r {
                        iterations_iternal.push(IterOptionInternal::Pruning(false));
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

        //let mut ref_file = std::io::BufReader::new(
        //std::fs::File::open(
        //"/home/ndokmai/workspace/phasing-oram/phasing/tests/ref_tprobs_chr1.bin",
        //)
        //.unwrap(),
        //);

        //let mut ref_file = std::io::BufWriter::new(
        //std::fs::File::create(
        //"/home/ndokmai/workspace/phasing-oram/phasing/tests/ref_tprobs_chr1.bin",
        //)
        //.unwrap(),
        //);

        let mut mcmc = Self::initialize(&params, genotypes.view());

        //mcmc.save_estimated_haps(&mut ref_file);
        //mcmc.check_estimated_haps(&mut ref_file);

        for i in iterations_iternal {
            mcmc.iteration(i, &mut rng);
            return mcmc.estimated_haps;

            //mcmc.save_tprobs(&mut ref_file);
            //mcmc.save_estimated_haps(&mut ref_file);
            //mcmc.check_tprobs(&mut ref_file);
            //mcmc.check_estimated_haps(&mut ref_file);
            //println!("{:#?}", mcmc.tprobs.slice(s![..10, .., ..]));
            //unimplemented!();
        }

        mcmc.phased_ind =
            viterbi::viterbi(mcmc.tprobs_dip.view(), mcmc.genotype_graph.graph.view());

        let first_ind = mcmc.phased_ind[0];
        let phased_ind_reduced = Array1::from_vec(
            mcmc.phased_ind
                .iter()
                .zip(mcmc.genotype_graph.graph.iter())
                .filter_map(|(&i, g)| if g.is_segment_marker() { Some(i) } else { None })
                .collect(),
        );

        mcmc.genotype_graph.traverse_graph_dip(
            first_ind,
            phased_ind_reduced.view(),
            mcmc.estimated_haps.view_mut(),
        );

        //mcmc.save_estimated_haps(&mut ref_file);
        //mcmc.check_estimated_haps(&mut ref_file);

        mcmc.estimated_haps
    }

    fn initialize_new(params: &'a McmcSharedParams, genotypes: ArrayView1<Genotype>) -> Self {
        println!("=== Initialization ===",);
        let now = Instant::now();
        let estimated_haps = initialize::initialize(&params.ref_panel, genotypes);
        let genotype_graph = GenotypeGraph::build(genotypes);
        let n_segments = genotype_graph
            .graph
            .iter()
            .filter(|v| v.is_segment_marker())
            .count();

        let first_ind = 0;
        let phased_ind = Array1::<u8>::zeros(n_segments);
        let first_tprobs_dip = Array1::<f64>::zeros(P);
        let tprobs_dip = Array3::<f64>::zeros((n_segments, P, P));
        let tprobs = Array3::<Real>::zeros((0, 0, 0));

        #[cfg(feature = "leak-resist-new")]
        let tprobs_e = Array2::<TpI8>::from_elem((estimated_haps.nrows(), P), TpI8::protect(0));

        let ignored_sites = Self::get_ignored_sites(params.variants.view(), genotypes);
        println!("Initialization: {} ms", (Instant::now() - now).as_millis());
        println!("",);
        Self {
            params,
            cur_overlap_region_len: params.overlap_region_len,
            genotype_graph,
            estimated_haps,
            first_ind,
            phased_ind,
            tprobs,
            first_tprobs_dip,
            tprobs_dip,
            ignored_sites,
            #[cfg(feature = "leak-resist-new")]
            tprobs_e,
        }
    }

    fn initialize(params: &'a McmcSharedParams, genotypes: ArrayView1<Genotype>) -> Self {
        println!("=== Initialization ===",);
        let now = Instant::now();
        let estimated_haps = initialize::initialize(&params.ref_panel, genotypes);
        let first_ind = 0;
        let phased_ind = Array1::<u8>::zeros(estimated_haps.nrows());
        let first_tprobs_dip = Array1::<f64>::zeros(0);
        let tprobs = Array3::<Real>::zeros((estimated_haps.nrows(), P, P));
        let tprobs_dip = Array3::<f64>::zeros((estimated_haps.nrows(), P, P));

        #[cfg(feature = "leak-resist-new")]
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
            first_ind,
            phased_ind,
            tprobs,
            first_tprobs_dip,
            tprobs_dip,
            ignored_sites,
            #[cfg(feature = "leak-resist-new")]
            tprobs_e,
        }
    }

    //fn initialize_from_input(
    //params: &'a McmcSharedParams,
    //genotypes: ArrayView1<Genotype>,
    //estimated_haps: Array2<Genotype>,
    //) -> Self {
    //println!("=== Initialization ===",);
    //let now = Instant::now();
    //let phased_ind = Array2::<u8>::zeros((estimated_haps.nrows(), 2));
    //let tprobs = Array3::<Real>::zeros((estimated_haps.nrows(), P, P));
    //let genotype_graph = GenotypeGraph::build(genotypes);
    //let ignored_sites = Self::get_ignored_sites(params.variants.view(), genotypes);
    //println!("Initialization: {} ms", (Instant::now() - now).as_millis());
    //println!("",);
    //Self {
    //params,
    //cur_overlap_region_len: params.overlap_region_len,
    //genotype_graph,
    //estimated_haps,
    //phased_ind,
    //tprobs,
    //ignored_sites,
    //}
    //}

    fn iteration_new(&mut self, iter_option: IterOptionInternalNew, mut rng: impl Rng) {
        {
            *EMIS_T.lock().unwrap() = Duration::from_millis(0);
            *TRAN_T.lock().unwrap() = Duration::from_millis(0);
            *COLL_T.lock().unwrap() = Duration::from_millis(0);
            *COMB_T.lock().unwrap() = Duration::from_millis(0);
            *COMBD_T.lock().unwrap() = Duration::from_millis(0);
            *PBWT_T.lock().unwrap() = Duration::from_millis(0);
            *EXPAND_T.lock().unwrap() = Duration::from_millis(0);
            *NEIGHBOR_T.lock().unwrap() = Duration::from_millis(0);
        }

        println!("=== {:?} Iteration ===", iter_option);
        let now = Instant::now();

        let max_k = None;

        if let Some(max_k) = max_k {
            println!("Max K: {max_k}");
        } else {
            println!("Max K: no upper limit");
        }
        let pbwt_group_filter = self.params.randomize_pbwt_group_bitmask(&mut rng);
        let windows = self.windows_full_segments(&mut rng);

        let mut prev_ind = None;
        let mut start_segment = 0;
        let mut sum_window_size = 0;
        let n_windows = windows.len();
        let mut ks = Vec::new();

        let mut is_first_window = true;

        let rprobs = self.params.hmm_params.get_rprobs(
            self.ignored_sites.view(),
            &self.genotype_graph,
            self.params.variants.view(),
        );

        let mut first_ind = None;

        for ((start_w, end_w), (start_write_w, end_write_w)) in windows.into_iter() {
            sum_window_size +=
                self.params.variants[end_w - 1].bp - self.params.variants[start_w].bp;

            let estimated_haps_w = self.estimated_haps.slice(s![start_w..end_w, ..]).to_owned();
            let params_w = self.params.slice(start_w, end_w);
            let pbwt_evaluted_filter_w = &self.params.pbwt_evaluted[start_w..end_w];
            let pbwt_group_filter_w = &pbwt_group_filter[start_w..end_w];
            let (selected_ref_panel, k) = select_ref_panel(
                &params_w.ref_panel,
                estimated_haps_w.view(),
                pbwt_evaluted_filter_w,
                pbwt_group_filter_w,
                self.params.s,
                max_k,
                Some(&mut rng),
            );
            ks.push(k as f64);

            let ignored_sites_w = self.ignored_sites.slice(s![start_w..end_w]);
            let rprobs_w = rprobs.slice(start_w, end_w);
            let genotype_graph_w = self.genotype_graph.slice(start_w, end_w);
            let n_segments_w = self
                .genotype_graph
                .slice(start_write_w, end_write_w)
                .graph
                .iter()
                .filter(|v| v.is_segment_marker())
                .count();

            let sampling_args = if iter_option.is_sample() {
                Some((
                    prev_ind,
                    &mut rng,
                    self.phased_ind
                        .slice_mut(s![start_segment..start_segment + n_segments_w]),
                ))
            } else {
                None
            };

            let target_tprobs_dip_w = if iter_option.is_save() {
                Some((
                    if is_first_window {
                        Some(self.first_tprobs_dip.view_mut())
                    } else {
                        None
                    },
                    self.tprobs_dip.slice_mut(s![
                        start_segment..start_segment + n_segments_w,
                        ..,
                        ..
                    ]),
                ))
            } else {
                None
            };

            let is_first_prune_main = iter_option.is_first_save();

            let first_ind_ = HmmNew::fwbw_sample(
                selected_ref_panel.view(),
                genotype_graph_w.graph,
                self.params.hmm_params.eprob,
                &rprobs_w,
                ignored_sites_w,
                is_first_window,
                sampling_args,
                target_tprobs_dip_w,
                is_first_prune_main,
            );

            if is_first_window {
                first_ind = first_ind_;
                if iter_option.is_sample() {
                    self.first_ind = first_ind_.unwrap();
                }
            }

            if iter_option.is_sample() {
                prev_ind = Some(self.phased_ind[start_segment + n_segments_w - 1]);
            }
            is_first_window = false;
            start_segment += n_segments_w;
        }

        if iter_option.is_sample() {
            self.genotype_graph.traverse_graph_dip(
                first_ind.unwrap(),
                self.phased_ind.view(),
                self.estimated_haps.view_mut(),
            );
        }

        //assert_eq!(start_segment, self.phased_ind.len());

        if iter_option == IterOptionInternalNew::Pruning(true) {
            self.genotype_graph.prune_rank_new(self.tprobs_dip.view());
            //self.cur_overlap_region_len *= 2;
        }

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
        println!("PBWT: {} ms", PBWT_T.lock().unwrap().as_millis());
        println!(
            "PBWT (expansion only): {} ms",
            EXPAND_T.lock().unwrap().as_millis()
        );
        println!(
            "Neighbor finding: {} ms",
            NEIGHBOR_T.lock().unwrap().as_millis()
        );
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

    fn iteration(&mut self, iter_option: IterOptionInternal, mut rng: impl Rng) {
        {
            *EMIS_T.lock().unwrap() = Duration::from_millis(0);
            *TRAN_T.lock().unwrap() = Duration::from_millis(0);
            *COLL_T.lock().unwrap() = Duration::from_millis(0);
            *COMB_T.lock().unwrap() = Duration::from_millis(0);
            *COMBD_T.lock().unwrap() = Duration::from_millis(0);
            *PBWT_T.lock().unwrap() = Duration::from_millis(0);
            *EXPAND_T.lock().unwrap() = Duration::from_millis(0);
            *NEIGHBOR_T.lock().unwrap() = Duration::from_millis(0);
        }

        println!("=== {:?} Iteration ===", iter_option);
        let now = Instant::now();

        #[cfg(feature = "leak-resist-new")]
        //let max_k = Some(MAX_K);
        let max_k = None;

        #[cfg(not(feature = "leak-resist-new"))]
        let max_k = None;

        if let Some(max_k) = max_k {
            println!("Max K: {max_k}");
        } else {
            println!("Max K: no upper limit");
        }
        let pbwt_group_filter = self.params.randomize_pbwt_group_bitmask(&mut rng);
        let windows = self.windows_full_segments(&mut rng);
        //let windows = self.windows(&mut rng);

        let mut prev_ind = 0;
        let mut sum_window_size = 0;
        let n_windows = windows.len();
        let mut ks = Vec::new();
        let mut tprob_pairs = Array2::<f64>::zeros((P, P));

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
            let estimated_haps_w = self.estimated_haps.slice(s![start_w..end_w, ..]).to_owned();
            let genotype_graph_w = self.genotype_graph.slice(start_w, end_w);
            let params_w = self.params.slice(start_w, end_w);
            let rprobs_w = rprobs.slice(start_w, end_w);
            let pbwt_evaluted_filter_w = &self.params.pbwt_evaluted[start_w..end_w];
            let pbwt_group_filter_w = &pbwt_group_filter[start_w..end_w];
            let ignored_sites_w = self.ignored_sites.slice(s![start_w..end_w]);
            let (selected_ref_panel, k) = select_ref_panel(
                &params_w.ref_panel,
                estimated_haps_w.view(),
                pbwt_evaluted_filter_w,
                pbwt_group_filter_w,
                self.params.s,
                max_k,
                Some(&mut rng),
            );
            //let (selected_ref_panel, k) = select_ref_panel_count(
            //&params_w.ref_panel,
            //estimated_haps_w.view(),
            //pbwt_evaluted_filter_w,
            //pbwt_group_filter_w,
            //self.params.s,
            //max_k,
            //);
            ks.push(k as f64);

            let mut hmm = Hmm::new();
            let mut tprobs_window = hmm.forward_backward(
                selected_ref_panel.view(),
                genotype_graph_w.graph.view(),
                &self.params.hmm_params,
                &rprobs_w,
                ignored_sites_w,
                is_first_window,
            );

            #[cfg(feature = "leak-resist-new")]
            let tprobs_e_window_src = hmm
                .tprobs_e
                .slice(s![start_write_w - start_w..end_write_w - start_w, .., ..])
                .to_owned();

            let tprobs_window_src =
                tprobs_window.slice_mut(s![start_write_w - start_w..end_write_w - start_w, .., ..]);

            #[cfg(not(feature = "leak-resist-new"))]
            let genotype_graph = &self.genotype_graph;

            let mut tprobs_window_target =
                self.tprobs_dip
                    .slice_mut(s![start_write_w..end_write_w, .., ..]);

            #[cfg(feature = "leak-resist-new")]
            let mut j = start_write_w - start_w;

            //// TODO: for debugging. Delete this
            //if iter_option == IterOptionInternal::Burnin {
            //Zip::from(tprobs_window_target.outer_iter_mut())
            //.and(tprobs_window_src.outer_iter())
            //.for_each(|mut a, b| {
            //#[cfg(feature = "leak-resist-new")]
            //{
            //a.assign(&crate::hmm::debug_expose_array(b, hmm.bprobs_e.row(j)));
            //j += 1;
            //}
            //#[cfg(not(feature = "leak-resist-new"))]
            //a.assign(&b);
            //});
            //}

            if iter_option == IterOptionInternal::Pruning(true)
                || iter_option == IterOptionInternal::Pruning(false)
            {
                Zip::from(tprobs_window_target.outer_iter_mut())
                    .and(tprobs_window_src.outer_iter())
                    .for_each(|a, b| {
                        #[cfg(feature = "leak-resist-new")]
                        {
                            hmm.cur_i = j;
                            j += 1;
                            hmm.combine_dips(b, a);
                        }

                        #[cfg(not(feature = "leak-resist-new"))]
                        if i == 0 || genotype_graph.graph[i].is_segment_marker() {
                            hmm.combine_dips(b, a);
                        } else {
                            let mut a = a;
                            a.fill(0.);
                        }
                        i += 1;
                    });
            }

            if let IterOptionInternal::Main(first_main) = iter_option {
                if first_main {
                    Zip::from(tprobs_window_target.outer_iter_mut())
                        .and(tprobs_window_src.outer_iter())
                        .for_each(|a, b| {
                            #[cfg(feature = "leak-resist-new")]
                            {
                                hmm.cur_i = j;
                                j += 1;
                                hmm.combine_dips(b, a);
                            }

                            #[cfg(not(feature = "leak-resist-new"))]
                            if i == 0 || genotype_graph.graph[i].is_segment_marker() {
                                hmm.combine_dips(b, a);
                            } else {
                                let mut a = a;
                                a.fill(0.);
                            }
                            i += 1;
                        });
                } else {
                    Zip::from(tprobs_window_target.outer_iter_mut())
                        .and(tprobs_window_src.outer_iter())
                        .for_each(|mut a, b| {
                            #[cfg(feature = "leak-resist-new")]
                            {
                                hmm.cur_i = j;
                                j += 1;
                                hmm.combine_dips(b, tprob_pairs.view_mut());
                                a += &tprob_pairs;
                            }

                            #[cfg(not(feature = "leak-resist-new"))]
                            if i == 0 || genotype_graph.graph[i].is_segment_marker() {
                                hmm.combine_dips(b, tprob_pairs.view_mut());
                                a += &tprob_pairs;
                            }

                            i += 1;
                        });
                }
            }

            // sample
            let phased_ind_window = sampling::forward_sampling(
                prev_ind,
                tprobs_window_src.view(),
                #[cfg(feature = "leak-resist-new")]
                tprobs_e_window_src.view(),
                genotype_graph_w
                    .graph
                    .slice(s![start_write_w - start_w..end_write_w - start_w]),
                is_first_window,
                &mut rng,
            );

            let mut tmp = self.phased_ind.slice_mut(s![start_write_w..end_write_w]);
            tmp.assign(&phased_ind_window);
            prev_ind = phased_ind_window[phased_ind_window.len() - 1];
            is_first_window = false;
        }

        let first_ind = self.phased_ind[0];
        let phased_ind_reduced = Array1::from_vec(
            self.phased_ind
                .iter()
                .zip(self.genotype_graph.graph.iter())
                .filter_map(|(&i, g)| if g.is_segment_marker() { Some(i) } else { None })
                .collect(),
        );

        self.genotype_graph.traverse_graph_dip(
            first_ind,
            phased_ind_reduced.view(),
            self.estimated_haps.view_mut(),
        );

        if iter_option == IterOptionInternal::Pruning(true)
            || iter_option == IterOptionInternal::Pruning(false)
        {
            self.genotype_graph.prune_rank(self.tprobs_dip.view());
            //self.genotype_graph.prune(self.tprobs.view());
            self.cur_overlap_region_len *= 2;
        }

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
        println!("PBWT: {} ms", PBWT_T.lock().unwrap().as_millis());
        println!(
            "PBWT (expansion only): {} ms",
            EXPAND_T.lock().unwrap().as_millis()
        );
        println!(
            "Neighbor finding: {} ms",
            NEIGHBOR_T.lock().unwrap().as_millis()
        );
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

    fn windows(&self, mut rng: impl Rng) -> Vec<((usize, usize), (usize, usize))> {
        let windows = windows_split::split(
            self.params.variants.view(),
            self.params.min_window_len_cm,
            &mut rng,
        );

        let overlap_len = self.cur_overlap_region_len;
        let mut hmm_windows = Vec::with_capacity(windows.len());
        let mut prev_end_write_boundary = 0;
        for (i, window) in windows.iter().enumerate() {
            let split_point = window.1;

            let mut end_write_boundary = None;
            for i in 0..overlap_len {
                if split_point + i >= self.genotype_graph.graph.len() {
                    break;
                }
                if self.genotype_graph.graph[split_point + i].is_segment_marker() {
                    end_write_boundary = Some(split_point + i);
                    break;
                }
            }
            let end_write_boundary = match end_write_boundary {
                Some(e) => e,
                None => split_point + overlap_len,
            };

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

    fn save_tprobs(&self, mut writer: impl std::io::Write) {
        //Zip::from(&self.genotype_graph.graph)
        //.and(self.tprobs.outer_iter())
        //.for_each(|g, t| {
        //if g.is_segment_marker() {
        //bincode::serialize_into(&mut writer, &t).unwrap();
        //}
        //});
    }

    fn check_tprobs(&self, mut reader: impl std::io::Read) {
        //Zip::indexed(&self.genotype_graph.graph)
        //.and(self.tprobs.outer_iter())
        //.for_each(|i, g, t| {
        //if g.is_segment_marker() {
        //let ref_t: Array2<Real> = bincode::deserialize_from(&mut reader).unwrap();
        ////if ref_t != t {
        ////println!("{i}:");
        ////assert_eq!(ref_t/t );
        ////}
        //const R: f64 = 0.1;
        //for (a, b) in ref_t.iter().zip(t.iter()) {
        //if a / b >= 1. + R || a / b < 1. - R {
        //println!("{i}:");
        //println!("{:#?}", ref_t);
        //println!("{:#?}", t);
        //println!("{a}, {b}, {}", a / b);
        //panic!();
        //}
        //}
        //}
        //});
    }

    fn save_estimated_haps(&self, writer: impl std::io::Write) {
        bincode::serialize_into(writer, &self.estimated_haps).unwrap();
    }

    fn check_estimated_haps(&self, reader: impl std::io::Read) {
        let ref_haps: Array2<i8> = bincode::deserialize_from(reader).unwrap();
        assert_eq!(self.estimated_haps, ref_haps);
    }

    fn windows_full_segments(&self, mut rng: impl Rng) -> Vec<((usize, usize), (usize, usize))> {
        let windows = windows_split::split_by_segment(
            &self.genotype_graph,
            self.params.variants.view(),
            self.params.min_window_len_cm,
            &mut rng,
        );
        //let windows = windows_split::split(
        //self.params.variants.view(),
        //self.params.min_window_len_cm,
        //&mut rng,
        //);
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
    ) -> Array1<BoolMcc> {
        let mut ignored_sites =
            Array1::<BoolMcc>::from_elem(variants.dim(), tp_value_new!(false, bool));
        Zip::from(&mut ignored_sites)
            .and(&variants)
            .and(&genotypes)
            .for_each(|s, v, &g| {
                let rarity = v.rarity();
                *s = tp_value_new!(
                    (g == 0 || g == 2)
                        && !(rarity == Rarity::NotRare
                            || (rarity == Rarity::Rare(true) && g == 2)
                            || (rarity == Rarity::Rare(false) && g == 0)),
                    bool
                );
            });
        ignored_sites
    }
}

fn select_ref_panel(
    ref_panel: &RefPanelSlice,
    estimated_haps: ArrayView2<Genotype>,
    pbwt_evaluted_filter: &[bool],
    pbwt_group_filter: &[bool],
    s: usize,
    max_k: Option<usize>,
    rng: Option<impl Rng>,
) -> (Array2<Genotype>, usize) {
    let n_pbwt_pos = pbwt_evaluted_filter.iter().filter(|b| **b).count();
    let neighbors_bitmap = neighbors_finding::find_neighbors(
        ref_panel
            .iter()
            .zip(pbwt_evaluted_filter.iter())
            .filter_map(|(v, &b)| if b { Some(v) } else { None }),
        estimated_haps
            .rows()
            .into_iter()
            .map(|r| r.to_owned())
            .zip(pbwt_evaluted_filter.iter())
            .filter_map(|(v, &b)| if b { Some(v) } else { None }),
        pbwt_group_filter
            .iter()
            .zip(pbwt_evaluted_filter.iter())
            .filter_map(|(&b1, &b2)| if b2 { Some(b1) } else { None }),
        n_pbwt_pos,
        ref_panel.n_haps,
        estimated_haps.ncols(),
        s,
    );

    #[cfg(feature = "leak-resist-new")]
    assert!(neighbors_bitmap.iter().filter(|&&b| b).count() <= MAX_K_OVERFLOW);

    let neighbors_bitmap = if let Some(max_k) = max_k {
        let k = neighbors_bitmap.iter().filter(|&&b| b).count();
        if k > max_k {
            println!("K > MAX_K: {} > {}", k, max_k);
        }
        let mut neighbors_bitmap_new = vec![false; neighbors_bitmap.len()];
        let neighbors_idx = neighbors_bitmap
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| if v { Some(i) } else { None })
            .collect::<Vec<_>>();
        let neighbors_idx = if k > max_k {
            use rand::prelude::SliceRandom;
            neighbors_idx
                .choose_multiple(&mut rng.unwrap(), max_k)
                .cloned()
                .collect::<Vec<_>>()
        } else {
            neighbors_idx
        };
        for i in neighbors_idx {
            neighbors_bitmap_new[i] = true;
        }
        neighbors_bitmap_new
    } else {
        neighbors_bitmap
    };

    //#[cfg(feature = "leak-resist")]
    //let bitmap = {
    //let mut bitmap = union_filter::OblivBitmap::new(nhap, oram_sgx::LinearScanningORAMCreator);
    //for i in neighbors.into_iter().flatten().flatten() {
    //bitmap.set(i);
    //}

    //bitmap
    //.into_iter()
    //.map(|v| tp_value!(v, bool))
    //.collect::<Vec<_>>()
    //};

    let k = neighbors_bitmap.iter().filter(|&&b| b).count();
    (ref_panel.filter(&neighbors_bitmap), k)
}

fn select_ref_panel_count(
    ref_panel: &RefPanelSlice,
    estimated_haps: ArrayView2<Genotype>,
    pbwt_evaluted_filter: &[bool],
    pbwt_group_filter: &[bool],
    s: usize,
    max_k: Option<usize>,
) -> (Array2<Genotype>, usize) {
    let n_pbwt_pos = pbwt_evaluted_filter.iter().filter(|b| **b).count();
    let neighbors_count = neighbors_finding::find_neighbors_count(
        ref_panel
            .iter()
            .zip(pbwt_evaluted_filter.iter())
            .filter_map(|(v, &b)| if b { Some(v) } else { None }),
        estimated_haps
            .rows()
            .into_iter()
            .map(|r| r.to_owned())
            .zip(pbwt_evaluted_filter.iter())
            .filter_map(|(v, &b)| if b { Some(v) } else { None }),
        pbwt_group_filter
            .iter()
            .zip(pbwt_evaluted_filter.iter())
            .filter_map(|(&b1, &b2)| if b2 { Some(b1) } else { None }),
        n_pbwt_pos,
        ref_panel.n_haps,
        estimated_haps.ncols(),
        s,
    );

    #[cfg(feature = "leak-resist-new")]
    debug_assert!(neighbors_count.iter().filter(|&&b| b > 0).count() <= MAX_K_OVERFLOW);

    let neighbors_bitmap = if let Some(max_k) = max_k {
        let k = neighbors_count.iter().filter(|&&b| b > 0).count();
        if k > max_k {
            println!("K > MAX_K: {} > {}", k, max_k);
            let mut neighbors_bitmap = vec![false; neighbors_count.len()];
            let mut neighbors_idx = neighbors_count
                .into_iter()
                .enumerate()
                .filter_map(|(i, v)| if v > 0 { Some((v, i)) } else { None })
                .collect::<Vec<_>>();
            neighbors_idx.sort();
            for (_, i) in neighbors_idx.into_iter().take(max_k) {
                neighbors_bitmap[i] = true;
            }
            neighbors_bitmap
        } else {
            neighbors_count.into_iter().map(|v| v > 0).collect()
        }
    } else {
        neighbors_count.into_iter().map(|v| v > 0).collect()
    };

    //#[cfg(feature = "leak-resist")]
    //let bitmap = {
    //let mut bitmap = union_filter::OblivBitmap::new(nhap, oram_sgx::LinearScanningORAMCreator);
    //for i in neighbors.into_iter().flatten().flatten() {
    //bitmap.set(i);
    //}

    //bitmap
    //.into_iter()
    //.map(|v| tp_value!(v, bool))
    //.collect::<Vec<_>>()
    //};

    let k = neighbors_bitmap.iter().filter(|&&b| b).count();
    (ref_panel.filter(&neighbors_bitmap), k)
}

#[cfg(test)]
mod test {}
