mod params;
mod initialize;
mod sampling;
mod viterbi;
pub use params::*;

use crate::genotype_graph::GenotypeGraph;
use crate::hmm::forward_backward;
use crate::neighbors_finding;
use crate::ref_panel::RefPanelSlice;
use crate::variants::{Rarity, Variant};
use crate::{Genotype, Real};
use rand::Rng;

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut3, Zip};
use std::time::Instant;

const N_HETS_PER_SEGMENT: usize = 3;
const P: usize = 1 << N_HETS_PER_SEGMENT;

#[derive(PartialEq, Eq, Debug)]
pub enum IterOption {
    Burnin,
    Pruning,
    Main(bool), // First main iteration?
}

pub struct Mcmc<'a> {
    pub params: &'a McmcSharedParams,
    pub cur_overlap_region_len: usize,
    pub genotype_graph: GenotypeGraph,
    pub ignored_sites: Array1<bool>,
    pub estimated_haps: Array2<Genotype>,
    phased_ind: Array2<u8>,
    tprobs: Array3<Real>,
}

impl<'a> Mcmc<'a> {
    pub fn initialize(params: &'a McmcSharedParams, genotypes: ArrayView1<Genotype>) -> Self {
        println!("=== Initialization ===",);
        let now = Instant::now();
        let estimated_haps = initialize::initialize(&params.ref_panel, genotypes);
        let phased_ind = unsafe { Array2::<u8>::uninit((estimated_haps.nrows(), 2)).assume_init() };
        let tprobs =
            unsafe { Array3::<Real>::uninit((estimated_haps.nrows(), P, P)).assume_init() };
        let genotype_graph = GenotypeGraph::build(genotypes);
        let ignored_sites = Self::get_ignored_sites(params.variants.view(), genotypes);
        println!(
            "#ignored_sites = {}",
            ignored_sites.iter().filter(|b| **b).count()
        );
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
            let ignored_sites_w = self.ignored_sites.slice(s![start_w..end_w]);
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
                params_w.variants,
                ignored_sites_w,
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
            renormalize(tprobs_window_slice.view_mut());

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

        self.phased_ind = viterbi::viterbi(self.tprobs.view());
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

    fn get_ignored_sites(
        variants: ArrayView1<Variant>,
        genotypes: ArrayView1<Genotype>,
    ) -> Array1<bool> {
        //Array1::<bool>::from_elem(variants.dim(), false)
        let mut ignored_sites = unsafe { Array1::<bool>::uninit(variants.dim()).assume_init() };
        Zip::from(&mut ignored_sites)
            .and(&variants)
            .and(&genotypes)
            .for_each(|s, v, &g| {
                let rarity = v.rarity();
                *s = (g == 0 || g == 2)
                    && !(rarity == Rarity::NotRare
                        || (rarity == Rarity::Rare(true) && g == 2)
                        || (rarity == Rarity::Rare(false) && g == 0));
            });
        ignored_sites
    }
}

#[inline]
fn renormalize(mut v: ArrayViewMut3<Real>) {
    Zip::from(v.outer_iter_mut()).for_each(|mut v_pos| {
        Zip::from(v_pos.outer_iter_mut()).for_each(|mut v_row| v_row /= v_row.sum())
    });
}

fn select_ref_panel(
    ref_panel: &RefPanelSlice,
    estimated_haps: ArrayView2<Genotype>,
    pbwt_filter_bitmask: &[bool],
    s: usize,
) -> Array2<Genotype> {
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

    println!("k = {}", neighbors_bitmap.iter().filter(|&&b| b).count());
    ref_panel.filter(&neighbors_bitmap)
}
