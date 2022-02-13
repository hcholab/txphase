use crate::genotype_graph::GenotypeGraph;
use crate::hmm::{forward_backward, HmmParams, HmmParamsSlice};
use crate::neighbors_finding;
use crate::ref_panel::{RefPanel, RefPanelSlice};
use crate::sampling;
use crate::{Genotype, Real};
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
    windows: Vec<(usize, usize)>,
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
        let windows = crate::windows_split::split(&cms, min_window_len_cm);
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
    pub cur_overlap_region_len: usize,
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
        Self {
            params,
            cur_overlap_region_len: params.overlap_region_len,
            genotype_graph,
            estimated_haps,
            phased_ind,
            tprobs,
        }
    }

    pub fn iteration(&mut self, iter_option: IterOption, mut rng: impl Rng) {
        println!("=== {:?} Iteration ===", iter_option);
        let now = Instant::now();

        let pbwt_filter_bitmask = self.params.randomize_pbwt_bitmask(&mut rng);
        let windows = self.overlap();

        let mut prev_ind = (0, 0);
        for ((start_w, end_w), (start_write_w, end_write_w)) in windows.into_iter() {
            let estimated_haps_w = self.estimated_haps.slice(s![start_w..end_w, ..]).to_owned();
            let genotype_graph_w = self.genotype_graph.slice(start_w, end_w);
            let params_w = self.params.slice(start_w, end_w);
            let pbwt_filter_bitmask_w = &pbwt_filter_bitmask[start_w..end_w];
            let selected_ref_panel = select_ref_panel(
                &params_w.ref_panel,
                estimated_haps_w.view(),
                pbwt_filter_bitmask_w,
                self.params.s,
            );

            let mut tprobs_window = forward_backward(
                selected_ref_panel.view(),
                genotype_graph_w.graph.view(),
                &params_w.hmm_params,
            );

            let mut tprobs_window_slice =
                tprobs_window.slice_mut(s![start_write_w - start_w..end_write_w - start_w, .., ..]);

            if iter_option == IterOption::Pruning {
                //save
                let mut tmp = self
                    .tprobs
                    .slice_mut(s![start_write_w..end_write_w, .., ..]);
                tmp.assign(&tprobs_window_slice);
            }

            // renormalize
            renormalize_pos_row(tprobs_window_slice.view_mut());

            if let IterOption::Main(first_main) = iter_option {
                let mut tmp = self
                    .tprobs
                    .slice_mut(s![start_write_w..end_write_w, .., ..]);
                if first_main {
                    tmp.assign(&tprobs_window_slice);
                } else {
                    tmp += &tprobs_window_slice
                }
            }

            // sample
            let phased_ind_window =
                sampling::forward_sampling(prev_ind, tprobs_window_slice.view(), &mut rng);
            let mut tmp = self
                .phased_ind
                .slice_mut(s![start_write_w..end_write_w, ..]);
            tmp.assign(&phased_ind_window);
            prev_ind = (
                phased_ind_window[[phased_ind_window.nrows() - 1, 0]],
                phased_ind_window[[phased_ind_window.nrows() - 1, 1]],
            );
        }

        self.genotype_graph
            .traverse_graph_pair(self.phased_ind.view(), self.estimated_haps.view_mut());

        if iter_option == IterOption::Pruning {
            self.genotype_graph.prune(self.tprobs.view());
            self.cur_overlap_region_len *= 2;
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
        self.genotype_graph
            .traverse_graph_pair(self.phased_ind.view(), self.estimated_haps.view_mut());
        self.estimated_haps
    }

    fn overlap(&self) -> Vec<((usize, usize), (usize, usize))> {
        let overlap_len = self.cur_overlap_region_len;
        let mut hmm_windows = Vec::with_capacity(self.params.windows.len());
        let mut prev_end_write_boundary = 0;
        for (i, window) in self.params.windows.iter().enumerate() {
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
            } else if i == self.params.windows.len() - 1 {
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
    //const N: usize = 1000;
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

    //println!("k = {}", neighbors_bitmap.iter().filter(|&&b| b).count());
    ref_panel.filter(&neighbors_bitmap)
}
