use crate::block;
use crate::genotype_graph::GenotypeGraph;
use crate::genotypes::GenotypesMeta;
use crate::neighbors_finding;
use crate::ref_panel::{RefPanel, RefPanelView};
use crate::sampling;
use crate::windows_split::Windows;
use crate::{tp_value, Genotype, Real};
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
    ref_panel: &'a RefPanel,
    genotypes_meta: &'a GenotypesMeta,
    windows: &'a Windows,
    eprob: f32,
    s: usize,
    capacity: usize,
    pub genotype_graph: GenotypeGraph,
    pub estimated_haps: Array2<Genotype>,
    phased_ind: Array2<u8>,
    tprobs: Array3<Real>,
}

impl<'a> Mcmc<'a> {
    pub fn initialize(
        ref_panel: &'a RefPanel,
        genotypes_meta: &'a GenotypesMeta,
        genotypes: ArrayView1<Genotype>,
        windows: &'a Windows,
        eprob: f32,
        s: usize,
        capacity: usize,
    ) -> Self {
        println!("=== Initialization ===",);
        let now = Instant::now();
        let estimated_haps = crate::initialize::initialize(&ref_panel, genotypes_meta, genotypes);
        let phased_ind = unsafe { Array2::<u8>::uninit((ref_panel.n_sites(), 2)).assume_init() };
        let tprobs = unsafe { Array3::<Real>::uninit((ref_panel.n_sites(), P, P)).assume_init() };
        let genotype_graph = GenotypeGraph::build(genotypes);
        println!("Initialization: {} ms", (Instant::now() - now).as_millis());
        println!("",);
        println!("",);
        Self {
            ref_panel,
            genotypes_meta,
            windows,
            eprob,
            s,
            capacity,
            genotype_graph,
            estimated_haps,
            phased_ind,
            tprobs,
        }
    }

    pub fn iteration(&mut self, iter_option: IterOption, mut rng: impl Rng) {
        println!("=== {:?} Iteration ===", iter_option);
        let now = Instant::now();

        let overlap = self.windows.n_pos_window_overlap;
        let mut prev_ind = (0, 0);

        let mut first = true;
        for &(start_w, end_w) in self.windows.boundaries.iter() {
            let ref_panel_w = self.ref_panel.sub_ref_panel(start_w, end_w);
            let estimated_haps_w = self.estimated_haps.slice(s![start_w..end_w, ..]).to_owned();
            let genotype_graph_w = self.genotype_graph.subview_mut(start_w, end_w);

            let hmm_params = HmmParams::init(&ref_panel_w, self.eprob);

            let selected_ref_panel =
                select_ref_panel(&ref_panel_w, estimated_haps_w.view(), self.s, self.capacity);

            let mut tprobs_window = genotype_graph_w.forward_pass(
                selected_ref_panel.view(),
                hmm_params.rprobs.view(),
                hmm_params.rev_rprobs.view(),
                hmm_params.eprob,
                hmm_params.rev_eprob,
            );

            if iter_option == IterOption::Pruning {
                //save
                if first {
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
                if first {
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
            if first {
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

            first = false;
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
    ref_panel: &RefPanelView,
    estimated_haps: ArrayView2<Genotype>,
    s: usize,
    _capacity: usize,
) -> Array2<Genotype> {
    const N: usize = 1000;
    let npos = ref_panel.n_sites();
    let nhap = ref_panel.meta.n_haps;
    let neighbors =
        neighbors_finding::find_neighbors(ref_panel.iter(), estimated_haps, npos, nhap, s);

    #[cfg(feature = "leak-resist")]
    let bitmap = {
        let mut bitmap = union_filter::OblivBitmap::new(nhap, oram_sgx::LinearScanningORAMCreator);
        for i in neighbors.into_iter().flatten().flatten() {
            bitmap.set(i);
        }

        bitmap
            .into_iter()
            .map(|v| tp_value!(v, bool))
            .collect::<Vec<_>>()
    };

    #[cfg(not(feature = "leak-resist"))]
    let bitmap = {
        let mut bitmap = vec![false; nhap];
        for i in neighbors.into_iter().flatten().flatten() {
            bitmap[i as usize] = true;
        }
        bitmap
    };

    println!("k = {}", bitmap.iter().filter(|&&b| b).count());

    let mut filtered_ref_panel = Array2::<Genotype>::from_elem((0, 0), tp_value!(0, i8));
    let mut cur_pos = 0;
    for (f, block) in ref_panel.blocks.iter().enumerate() {
        let nskip = if f == 0 { 0 } else { 1 };
        let nvar = block.nvar - nskip;
        let transposed = block::block_to_aligned_transposed::<N>(&block, nhap);

        #[cfg(feature = "leak-resist")]
        let (filtered, n_filtered) =
            oram_sgx::obliv_filter(&bitmap[..], &transposed, capacity as u32);

        #[cfg(feature = "leak-resist")]
        println!("n_filtered: {}", n_filtered.expose());

        #[cfg(not(feature = "leak-resist"))]
        let filtered = transposed
            .into_iter()
            .zip(bitmap.iter())
            .filter(|(_, b)| **b)
            .map(|(v, _)| v)
            //.take(capacity)
            .collect::<Vec<_>>();

        if filtered_ref_panel.ncols() != filtered.len() {
            filtered_ref_panel = Array2::<Genotype>::from_elem(
                (ref_panel.sites_bitmask.len(), filtered.len()),
                tp_value!(0, i8),
            );
        }

        // transpose
        for (i, hap) in filtered.into_iter().enumerate() {
            for (mut row, &geno) in filtered_ref_panel
                .slice_mut(s![cur_pos..(cur_pos + nvar), ..])
                .rows_mut()
                .into_iter()
                .zip(hap.as_slice().iter().skip(nskip).take(nvar))
            {
                row[i] = tp_value!(geno, i8);
            }
        }
        cur_pos += nvar;
    }
    let mut out =
        Array2::<Genotype>::from_elem((npos, filtered_ref_panel.ncols()), tp_value!(0, i8));

    let mut i = 0;
    for (row, b) in filtered_ref_panel
        .rows()
        .into_iter()
        .zip(ref_panel.sites_bitmask.iter())
    {
        if *b {
            out.slice_mut(s![i, ..]).assign(&row);
            i += 1;
        }
    }

    out
}

pub struct HmmParams {
    pub eprob: Real,
    pub rev_eprob: Real,
    pub rprobs: Array1<Real>,
    pub rev_rprobs: Array1<Real>,
}

impl HmmParams {
    pub fn init(ref_panel: &RefPanelView, eprob: f32) -> Self {
        let rev_eprob = 1. - eprob;
        #[cfg(feature = "leak-resist")]
        let (eprob, rev_eprob) = (Real::protect_f32(eprob), Real::protect_f32(rev_eprob));
        let rprobs = Array1::from_vec(ref_panel.recomb_probs.to_owned());
        let rev_rprobs = Array1::from_vec(ref_panel.rev_recomb_probs.to_owned());
        Self {
            eprob: eprob.into(),
            rev_eprob: rev_eprob.into(),
            rprobs,
            rev_rprobs,
        }
    }
}
