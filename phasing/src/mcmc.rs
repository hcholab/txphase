use crate::genotype_graph::GenotypeGraph;
use crate::hmm::{HmmParams, HmmParamsSlice};
use crate::neighbors_finding;
//use crate::ref_panel::{RefPanel, RefPanelSlice};
use crate::ref_panel::{RefPanel, RefPanelSlice};
use crate::sampling;
use crate::windows_split::Windows;
use crate::{tp_value, Genotype, Real};
use rand::Rng;

use ndarray::{s, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut3, Zip};
use std::time::Instant;

const N_HETS_PER_SEGMENT: usize = 3;
const P: usize = 1 << N_HETS_PER_SEGMENT;

#[derive(PartialEq, Eq, Debug)]
pub enum IterOption {
    Burnin,
    Pruning,
    Main(bool), // First main iteration?
}

pub struct McmcSharedParams {
    ref_panel: RefPanel,
    hmm_params: HmmParams,
    windows: Windows,
    overlap_region_len: usize,
    pbwt_groups: Vec<Vec<usize>>,
    s: usize,
}

impl McmcSharedParams {
    pub fn new(
        ref_panel: RefPanel,
        cms: Vec<f32>,
        min_window_len_cm: f32,
        overlap_region_len: usize,
        pbwt_modulo: f32,
        s: usize,
    ) -> Self {
        let hmm_params = HmmParams::new(&cms, ref_panel.n_haps);
        let windows = Windows::new(&cms, min_window_len_cm);
        let pbwt_groups = Self::pbwt_groups(&cms, pbwt_modulo);
        println!("#pbwt_positions = {}", pbwt_groups.len());
        Self {
            ref_panel,
            hmm_params,
            windows,
            overlap_region_len,
            pbwt_groups,
            s,
        }
    }

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> McmcSharedParamsSlice<'a> {
        McmcSharedParamsSlice {
            ref_panel: self.ref_panel.slice(start, end),
            hmm_params: self.hmm_params.slice(start, end),
        }
    }

    pub fn n_sites(&self) -> usize {
        self.hmm_params.rprobs.len()
    }

    pub fn randomize_pbwt_bitmask(&self, mut rng: impl Rng) -> Vec<bool> {
        use rand::prelude::SliceRandom;
        let mut pbwt_filter_bitmask = vec![false; self.n_sites()];
        for group in &self.pbwt_groups {
            let i = group.choose(&mut rng).unwrap();
            pbwt_filter_bitmask[*i] = true;
        }
        pbwt_filter_bitmask
    }

    fn pbwt_groups(cms: &[f32], pbwt_modulo: f32) -> Vec<Vec<usize>> {
        let mut start_cm = cms[0];
        let mut end_cm = start_cm + pbwt_modulo;
        let mut pbwt_groups = Vec::new();

        let mut cur_group = Vec::new();
        for (i, &cm) in cms.iter().enumerate() {
            if cm >= start_cm && cm < end_cm {
                cur_group.push(i);
            } else {
                if !cur_group.is_empty() {
                    pbwt_groups.push(cur_group);
                    cur_group = Vec::new();
                }
                while cm >= end_cm {
                    start_cm += pbwt_modulo;
                    end_cm += pbwt_modulo;
                }
            }
        }
        pbwt_groups
    }
}

pub struct McmcSharedParamsSlice<'a> {
    pub ref_panel: RefPanelSlice<'a>,
    pub hmm_params: HmmParamsSlice<'a>,
}

pub struct Mcmc<'a> {
    pub params: &'a McmcSharedParams,
    pub genotype_graph: GenotypeGraph,
    pub estimated_haps: Array2<Genotype>,
    phased_ind: Array2<u8>,
    tprobs: Array3<Real>,
}

impl<'a> Mcmc<'a> {
    pub fn initialize(params: &'a McmcSharedParams, genotypes: ArrayView1<Genotype>) -> Self {
        println!("=== Initialization ===",);
        let now = Instant::now();
        let estimated_haps = crate::initialize::initialize(&params.ref_panel, genotypes);
        let phased_ind = unsafe { Array2::<u8>::uninit((estimated_haps.nrows(), 2)).assume_init() };
        let tprobs =
            unsafe { Array3::<Real>::uninit((estimated_haps.nrows(), P, P)).assume_init() };
        let genotype_graph = GenotypeGraph::build(genotypes);
        println!("Initialization: {} ms", (Instant::now() - now).as_millis());
        println!("",);
        println!("",);
        Self {
            params,
            genotype_graph,
            estimated_haps,
            phased_ind,
            tprobs,
        }
    }

    pub fn iteration(&mut self, iter_option: IterOption, mut rng: impl Rng) {
        println!("=== {:?} Iteration ===", iter_option);
        let now = Instant::now();

        let overlap = self.params.overlap_region_len;

        let pbwt_filter_bitmask = self.params.randomize_pbwt_bitmask(&mut rng);

        let mut prev_ind = (0, 0);
        for (i, &(start_w, end_w)) in self.params.windows.boundaries.iter().enumerate() {
            //let (start_overlap, end_overlap) = if i == 0 {
            //(start_w, end_w + overlap)
            //} else if i == self.params.windows.boundaries.len() - 1 {
            //(start_w, end_w)
            //} else {
            //(start_w - overlap, end_w + overlap)
            //};

            // expand by overlap region len
            let (start_w, end_w) = if i == 0 {
                (start_w, end_w + overlap)
            } else if i == self.params.windows.boundaries.len() - 1 {
                (start_w, end_w)
            } else {
                (start_w, end_w + overlap)
            };

            let estimated_haps_w = self.estimated_haps.slice(s![start_w..end_w, ..]).to_owned();
            let genotype_graph_w = self.genotype_graph.subview_mut(start_w, end_w);
            let params_w = self.params.slice(start_w, end_w);
            let pbwt_filter_bitmask_w = &pbwt_filter_bitmask[start_w..end_w];
            let selected_ref_panel = select_ref_panel(
                &params_w.ref_panel,
                estimated_haps_w.view(),
                pbwt_filter_bitmask_w,
                self.params.s,
            );

            let mut tprobs_window =
                genotype_graph_w.forward_backward(selected_ref_panel.view(), &params_w.hmm_params);

            if iter_option == IterOption::Pruning {
                //save
                if i == 0 {
                    let mut tmp = self.tprobs.slice_mut(s![..end_w, .., ..]);
                    tmp.assign(&tprobs_window.view());
                } else {
                    let mut tmp = self.tprobs.slice_mut(s![start_w + overlap..end_w, .., ..]);
                    tmp.assign(&tprobs_window.slice(s![overlap.., .., ..]));
                }
            }

            // renormalize
            renormalize_pos_row(tprobs_window.view_mut());

            if let IterOption::Main(first_main) = iter_option {
                //add
                if i == 0 {
                    let mut tmp = self.tprobs.slice_mut(s![..end_w, .., ..]);
                    if first_main {
                        tmp.assign(&tprobs_window);
                    } else {
                        tmp += &tprobs_window.view();
                    }
                } else {
                    let mut tmp = self.tprobs.slice_mut(s![start_w + overlap..end_w, .., ..]);
                    if first_main {
                        tmp.assign(&tprobs_window.slice(s![overlap.., .., ..]));
                    } else {
                        tmp += &tprobs_window.slice(s![overlap.., .., ..]);
                    }
                }
            }

            // sample
            if i == 0 {
                let phased_ind_window =
                    sampling::forward_sampling(prev_ind, tprobs_window.view(), &mut rng);
                let mut tmp = self.phased_ind.slice_mut(s![..end_w, ..]);
                tmp.assign(&phased_ind_window);
                prev_ind = (
                    phased_ind_window[[phased_ind_window.nrows() - 1, 0]],
                    phased_ind_window[[phased_ind_window.nrows() - 1, 1]],
                );
            } else {
                let tprobs_window = tprobs_window.slice(s![overlap.., .., ..]);
                let phased_ind_window =
                    sampling::forward_sampling(prev_ind, tprobs_window, &mut rng);
                let mut tmp = self.phased_ind.slice_mut(s![start_w + overlap..end_w, ..]);
                tmp.assign(&phased_ind_window);
                prev_ind = (
                    phased_ind_window[[phased_ind_window.nrows() - 1, 0]],
                    phased_ind_window[[phased_ind_window.nrows() - 1, 1]],
                );
            }
        }

        let estimated_haps = self.genotype_graph.get_haps(self.phased_ind.view());
        self.estimated_haps = estimated_haps;

        if iter_option == IterOption::Pruning {
            self.genotype_graph.prune(self.tprobs.view());
        }

        println!("Iteration: {} ms", (Instant::now() - now).as_millis());
        println!("",);
    }

    pub fn main_finalize(mut self, n_main_rounds: usize, mut rng: impl Rng) -> Array2<Genotype> {
        assert!(n_main_rounds > 0);

        self.iteration(IterOption::Main(true), &mut rng);
        for _ in 0..n_main_rounds - 1 {
            self.iteration(IterOption::Main(false), &mut rng);
        }

        self.phased_ind = crate::viterbi::viterbi(self.tprobs.view());
        self.genotype_graph.get_haps(self.phased_ind.view())
    }

    //pub fn overlap_boundary(
    //start_w: usize,
    //end_w: usize,
    //overlap_region_len: usize,
    //is_first: bool,
    //is_last: bool,
    //segment_start_bitmask: &[bool],
    //) -> (usize, usize, bool) {

    //todo!()
    //}
}

#[inline]
fn renormalize_pos_row(mut v: ArrayViewMut3<Real>) {
    Zip::from(v.outer_iter_mut()).for_each(|mut v_pos| {
        Zip::from(v_pos.outer_iter_mut()).for_each(|mut v_row| v_row /= v_row.sum())
    });
}

#[inline]
fn renormalize_pos(mut v: ArrayViewMut3<Real>) {
    Zip::from(v.outer_iter_mut()).for_each(|mut v_pos| v_pos /= v_pos.sum());
}

fn select_ref_panel(
    ref_panel: &RefPanelSlice,
    estimated_haps: ArrayView2<Genotype>,
    pbwt_filter_bitmask: &[bool],
    s: usize,
) -> Array2<Genotype> {
    const N: usize = 1000;
    let n_pbwt_pos = pbwt_filter_bitmask.iter().filter(|b| **b).count();
    let neighbors_bitmap = neighbors_finding::find_neighbors(
        ref_panel
            .iter()
            .zip(pbwt_filter_bitmask.iter())
            .filter(|(_, b)| **b)
            .map(|(v, _)| v),
        estimated_haps
            .rows()
            .into_iter()
            .map(|r| r.to_owned())
            .zip(pbwt_filter_bitmask.iter())
            .filter(|(_, b)| **b)
            .map(|(v, _)| v),
        n_pbwt_pos,
        ref_panel.n_haps,
        estimated_haps.ncols(),
        s,
    );

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

    //#[cfg(not(feature = "leak-resist"))]
    //let bitmap = {
    //let mut bitmap = vec![false; nhap];
    //for i in neighbors.into_iter() {
    //bitmap[i as usize] = true;
    //}
    //bitmap
    //};

    println!("k = {}", neighbors_bitmap.iter().filter(|&&b| b).count());

    let mut filtered_ref_panel = Array2::<Genotype>::from_elem((0, 0), tp_value!(0, i8));
    let mut cur_pos = 0;
    for block in ref_panel.blocks.iter() {
        //let nskip = if f == 0 { 0 } else { 1 };
        let nvar = block.n_sites();
        let transposed =
            crate::ref_panel::block_to_aligned_transposed::<N>(&block, ref_panel.n_haps);

        //#[cfg(feature = "leak-resist")]
        //let (filtered, n_filtered) =
        //oram_sgx::obliv_filter(&[..], &transposed, capacity as u32);

        //#[cfg(feature = "leak-resist")]
        //println!("n_filtered: {}", n_filtered.expose());

        //#[cfg(not(feature = "leak-resist"))]
        let filtered = transposed
            .into_iter()
            .zip(neighbors_bitmap.iter())
            .filter(|(_, b)| **b)
            .map(|(v, _)| v)
            //.take(capacity)
            .collect::<Vec<_>>();

        if filtered_ref_panel.ncols() != filtered.len() {
            filtered_ref_panel = Array2::<Genotype>::from_elem(
                (ref_panel.n_sites, filtered.len()),
                tp_value!(0, i8),
            );
        }

        // transpose
        for (i, hap) in filtered.into_iter().enumerate() {
            for (mut row, &geno) in filtered_ref_panel
                .slice_mut(s![cur_pos..(cur_pos + nvar), ..])
                .rows_mut()
                .into_iter()
                .zip(hap.as_slice().iter())
            {
                row[i] = tp_value!(geno, i8);
            }
        }
        cur_pos += nvar;
    }
    filtered_ref_panel
    //let mut out = Array2::<Genotype>::from_elem(
    //(estimated_haps.nrows(), filtered_ref_panel.ncols()),
    //tp_value!(0, i8),
    //);

    //let mut i = 0;
    //for (row, b) in filtered_ref_panel
    //.rows()
    //.into_iter()
    //.zip(ref_panel.sites_bitmask.iter())
    //{
    //if *b {
    //out.slice_mut(s![i, ..]).assign(&row);
    //i += 1;
    //}
    //}

    //out
}
