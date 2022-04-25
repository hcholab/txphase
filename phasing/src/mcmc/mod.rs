mod initialize;
mod params;
mod sampling;
mod viterbi;
mod windows_split;
pub use params::*;

use crate::genotype_graph::GenotypeGraph;
use crate::hmm::Hmm;
use crate::hmm::{COLL_T, COMB_T, COMBD_T, EMIS_T, TRAN_T};
use crate::neighbors_finding;
use crate::ref_panel::RefPanelSlice;
use crate::variants::{Rarity, Variant};
use crate::{tp_value_new, BoolMcc, Genotype, Real};
use rand::Rng;

use std::time::{Duration, Instant};

#[cfg(feature = "leak-resist-new")]
use timing_shield::TpI8;

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Zip};

const N_HETS_PER_SEGMENT: usize = 3;
const P: usize = 1 << N_HETS_PER_SEGMENT;
const MAX_K: usize = 1000;

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
    ignored_sites: Array1<BoolMcc>,
    estimated_haps: Array2<Genotype>,
    phased_ind: Array2<u8>,
    tprobs: Array3<Real>,
    #[cfg(feature = "leak-resist-new")]
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

        //let mut ref_file = std::io::BufReader::new(
        //std::fs::File::open(
        //"/home/ndokmai/workspace/phasing-oram/phasing/tests/ref_tprobs_chr20.bin",
        //)
        //.unwrap(),
        //);

        //let mut ref_file = std::io::BufWriter::new(
        //std::fs::File::create(
        //"/home/ndokmai/workspace/phasing-oram/phasing/tests/ref_tprobs_chr20.bin",
        //)
        //.unwrap(),
        //);

        let mut mcmc = Self::initialize(&params, genotypes.view());

        //mcmc.save_estimated_haps(&mut ref_file);
        //mcmc.check_estimated_haps(&mut ref_file);

        for i in iterations_iternal {
            mcmc.iteration(i, &mut rng);
            //mcmc.save_tprobs(&mut ref_file);
            //mcmc.save_estimated_haps(&mut ref_file);
            //mcmc.check_tprobs(&mut ref_file);
            //mcmc.check_estimated_haps(&mut ref_file);
            //println!("{:#?}", mcmc.tprobs.slice(s![..10, .., ..]));
            //unimplemented!();
        }

        //#[cfg(feature = "leak-resist-new")]
        //let tprobs = Array3::<Real>::from_shape_fn(mcmc.tprobs.dim(), |(i, j, l)|
        //crate::hmm::debug_expose(mcmc.tprobs[[i, j, l]], self.bprobs_e[[i, j]])
        //);

        //#[cfg(not(feature = "leak-resist-new"))]
        //let tprobs = mcmc.tprobs.view();

        mcmc.phased_ind = viterbi::viterbi(mcmc.tprobs.view(), mcmc.genotype_graph.graph.view());

        mcmc.genotype_graph
            .traverse_graph_pair(mcmc.phased_ind.view(), mcmc.estimated_haps.view_mut());

        //mcmc.save_estimated_haps(&mut ref_file);
        //mcmc.check_estimated_haps(&mut ref_file);

        mcmc.estimated_haps
    }

    fn initialize(params: &'a McmcSharedParams, genotypes: ArrayView1<Genotype>) -> Self {
        println!("=== Initialization ===",);
        let now = Instant::now();
        let estimated_haps = initialize::initialize(&params.ref_panel, genotypes);
        let phased_ind = Array2::<u8>::zeros((estimated_haps.nrows(), 2));
        let tprobs = Array3::<Real>::zeros((estimated_haps.nrows(), P, P));

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
            phased_ind,
            tprobs,
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

        //let max_k = Some(MAX_K);
        let max_k = None;
        if let Some(max_k) = max_k {
            println!("Max K: {max_k}");
        } else {
            println!("Max K: no upper limit");
        }
        let pbwt_group_filter = self.params.randomize_pbwt_group_bitmask(&mut rng);
        let windows = self.windows_full_segments(&mut rng);
        //let windows = self.windows(&mut rng);

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
            );
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

            let tprobs_window_src =
                tprobs_window.slice_mut(s![start_write_w - start_w..end_write_w - start_w, .., ..]);
            #[cfg(not(feature = "leak-resist-new"))]
            let genotype_graph = &self.genotype_graph;

            let mut tprobs_window_target =
                self.tprobs
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

            if iter_option == IterOptionInternal::Pruning {
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
                genotype_graph_w
                    .graph
                    .slice(s![start_write_w - start_w..end_write_w - start_w]),
                is_first_window,
                &mut rng,
            );

            let mut tmp = self
                .phased_ind
                .slice_mut(s![start_write_w..end_write_w, ..]);
            tmp.assign(&phased_ind_window);
            prev_ind = (
                phased_ind_window[[phased_ind_window.nrows() - 1, 0]],
                phased_ind_window[[phased_ind_window.nrows() - 1, 1]],
            );
            is_first_window = false;
        }

        self.genotype_graph
            .traverse_graph_pair(self.phased_ind.view(), self.estimated_haps.view_mut());

        if iter_option == IterOptionInternal::Pruning {
            self.genotype_graph.prune_rank(self.tprobs.view());
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
        println!("Emission: {} ms", EMIS_T.lock().unwrap().as_millis());
        println!("Transition: {} ms", TRAN_T.lock().unwrap().as_millis());
        println!("Collapse: {} ms", COLL_T.lock().unwrap().as_millis());
        println!("Combine: {} ms", COMB_T.lock().unwrap().as_millis());
        println!("Combine Diploid: {} ms", COMBD_T.lock().unwrap().as_millis());
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
        Zip::from(&self.genotype_graph.graph)
            .and(self.tprobs.outer_iter())
            .for_each(|g, t| {
                if g.is_segment_marker() {
                    bincode::serialize_into(&mut writer, &t).unwrap();
                }
            });
    }

    fn check_tprobs(&self, mut reader: impl std::io::Read) {
        Zip::indexed(&self.genotype_graph.graph)
            .and(self.tprobs.outer_iter())
            .for_each(|i, g, t| {
                if g.is_segment_marker() {
                    assert!(t.into_iter().map(|&v| v != 0.).fold(true, |a, b| a || b));
                    let ref_t: Array2<Real> = bincode::deserialize_from(&mut reader).unwrap();
                    if ref_t != t {
                        println!("{i}:");
                        assert_eq!(ref_t, t);
                    }
                    //for (a, b) in ref_t.iter().zip(t.iter()) {
                    //if (a-b).abs() >= 1e-4 {
                    //println!("{a}, {b}, {}", (a-b).abs());
                    //}
                    //assert!((a-b).abs() < 1e-4);
                    //}
                }
            });
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

    let neighbors_bitmap = if let Some(max_k) = max_k {
        let mut neighbors_bitmap_new = vec![false; neighbors_bitmap.len()];
        let neighbors_idx = neighbors_bitmap
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| if v { Some(i) } else { None })
            .take(max_k)
            .collect::<Vec<_>>();
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

#[cfg(test)]
mod test {}
