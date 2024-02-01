use crate::hmm::params::*;

use crate::genotype_graph::{G, P};
use crate::inner::*;
use crate::rss_hmm::filtered_block::FilteredBlockSliceObliv;
use crate::tp_value;
use ndarray::{
    s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMut3, Zip,
};

#[cfg(feature = "obliv")]
use crate::dynamic_fixed::*;

#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::{TpEq, TpI16, TpI8, TpOrd};

#[cfg(feature = "obliv")]
use tp_fixedpoint::Dot;

#[cfg(not(feature = "obliv"))]
use ndarray::linalg::Dot;

#[cfg(not(feature = "obliv"))]
const RENORM_THRESHOLD: f64 = 1e-20;

use std::time::{Duration, Instant};

use std::cell::RefCell;
thread_local! {
    pub static EMISS: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static TRANS: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static COLL: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static COMB1: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static COMB2: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static EXPAND: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static BLOCK: RefCell<Duration> = RefCell::new(Duration::ZERO);
}

pub struct ProbBlock {
    pub c_prob: Array3<Real>,
    pub cnr_prob: Array3<Real>,
    #[cfg(feature = "obliv")]
    pub prob_e: Array2<TpI16>,
    pub alpha_pre: Array2<Real>,
    pub alpha_post: Array1<Real>,
    pub is_pre: Array1<bool>,
    pub save_alpha_post_cnr: Array2<Real>,
}

impl ProbBlock {
    pub fn new(n_sites: usize, n_unique: usize, n_haps: usize) -> Self {
        Self {
            c_prob: Array3::<Real>::zeros((n_sites, P, n_unique)),
            cnr_prob: Array3::<Real>::zeros((n_sites, P, n_unique)),
            #[cfg(feature = "obliv")]
            prob_e: Array2::<TpI16>::from_elem((n_sites, P), TpI16::protect(0)),
            alpha_pre: Array2::<Real>::zeros((n_haps, P)),
            alpha_post: Array1::<Real>::zeros(n_haps),
            is_pre: Array1::from_elem(n_sites, true),
            save_alpha_post_cnr: Array2::<Real>::zeros((P, n_unique)),
        }
    }

    //pub fn expand(
    //&self,
    //index_map: ArrayView1<u16>,
    //inv_weights: ArrayView1<Real>,
    //) -> Array3<Real> {
    //let mut expanded = Array3::<Real>::zeros((
    //self.n_segments(),
    //self.alpha_pre.dim().0,
    //self.alpha_pre.dim().1,
    //));
    //let cr_prob = &self.c_prob - &self.cnr_prob;
    //Zip::from(expanded.outer_iter_mut())
    //.and(cr_prob.outer_iter())
    //.and(self.cnr_prob.outer_iter())
    //.and(&self.is_pre)
    //.map_collect(|exp, cr, cnr, &is_pre| {
    //HmmReduced::expand_prob(
    //index_map,
    //inv_weights,
    //cr,
    //cnr,
    //self.alpha_pre.view(),
    //self.alpha_post.view(),
    //is_pre,
    //exp,
    //);
    //});
    //expanded
    //}

    pub fn n_sites(&self) -> usize {
        self.is_pre.len()
    }
}

pub struct ForwardProbSave<'a> {
    pub c: ArrayViewMut2<'a, Real>,
    pub cnr: ArrayViewMut2<'a, Real>,
    #[cfg(feature = "obliv")]
    pub e: ArrayViewMut1<'a, TpI16>,
}

pub struct HmmReduced {}

#[cfg(feature = "obliv")]
type FwbwOut = (Array1<Real>, Array1<TpI16>, Array3<Real>, Array3<TpI16>);

#[cfg(not(feature = "obliv"))]
type FwbwOut = (Array1<Real>, Array3<Real>);

impl HmmReduced {
    pub fn fwbw<'a>(
        blocks: &[FilteredBlockSliceObliv<'a>],
        n_sites_window: usize,
        n_full_states: UInt,
        n_full_haps: usize,
        genotype_graph: ArrayView1<G>,
        eprob: Real,
        rprobs: &RprobsSlice,
        _ignored_sites: ArrayView1<Bool>,
    ) -> FwbwOut {
        let _ignored_sites =
            Array1::<Bool>::from_elem(_ignored_sites.dim(), tp_value!(false, bool));
        let _ignored_sites = _ignored_sites.view();

        let bprobs = Self::backward(
            blocks,
            n_sites_window,
            n_full_states,
            n_full_haps,
            genotype_graph,
            eprob,
            rprobs,
            _ignored_sites,
        );

        let first_tprobs =
            Zip::from(bprobs[0].c_prob.slice(s![0, .., ..]).rows()).map_collect(|r| r.sum());

        #[cfg(feature = "obliv")]
        let first_tprobs_e = bprobs[0].prob_e.row(0).to_owned();

        let mut tprobs = Array3::<Real>::zeros((n_sites_window, P, P));

        #[cfg(feature = "obliv")]
        let mut tprobs_e = Array3::<TpI16>::from_elem((n_sites_window, P, P), TpI16::protect(0));

        #[cfg(feature = "obliv")]
        let mut rprobs_iter = rprobs.get_forward(n_full_states.as_u64());

        #[cfg(not(feature = "obliv"))]
        let mut rprobs_iter = rprobs.get_forward(n_full_states as usize);

        let mut site_i = 0;
        let mut is_first_segment = true;

        let mut cur_c_prob;

        #[cfg(feature = "obliv")]
        let mut cur_c_prob_e;

        let mut cur_cnr_prob;

        #[cfg(feature = "obliv")]
        let mut cur_cnr_prob_e;

        let mut prev_c_prob = Array2::<Real>::zeros((0, 0));

        #[cfg(feature = "obliv")]
        let mut prev_c_prob_e = Array1::from_elem(P, TpI16::protect(0));

        let mut prev_cnr_prob = Array2::<Real>::zeros((0, 0));

        #[cfg(feature = "obliv")]
        let mut prev_cnr_prob_e = Array1::from_elem(P, TpI16::protect(0));

        let mut full_trans_prob = Array2::<Real>::zeros((n_full_haps, P));

        #[cfg(feature = "obliv")]
        let mut full_trans_prob_e = Array1::from_elem(P, TpI16::protect(0));

        let mut prev_block_prob: Option<ProbBlock> = None;

        for (block_i, block) in blocks.iter().enumerate() {
            let block_n_sites = block.n_sites();
            let block_n_unique_haps = block.n_unique_haps();

            cur_c_prob = Array2::<Real>::zeros((P, block_n_unique_haps));
            cur_cnr_prob = Array2::<Real>::zeros((P, block_n_unique_haps));

            #[cfg(feature = "obliv")]
            {
                cur_c_prob_e = Array1::from_elem(P, TpI16::protect(0));
                cur_cnr_prob_e = Array1::from_elem(P, TpI16::protect(0));
            }

            let mut cur_block_prob =
                ProbBlock::new(block_n_sites, block_n_unique_haps, n_full_haps);

            if block_i == 0 {
                //let mut cur_c_prob = cur_block_prob.c_prob.slice_mut(s![0, .., ..]);

                //#[cfg(feature = "obliv")]
                //let cur_c_prob_e = cur_block_prob.prob_e.slice_mut(s![0, ..]);

                //let mut cur_cnr_prob = cur_block_prob.cnr_prob.slice_mut(s![0, .., ..]);

                Self::init(
                    true,
                    block,
                    genotype_graph[0],
                    eprob,
                    cur_c_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    cur_c_prob_e.view_mut(),
                    cur_block_prob.alpha_pre.view_mut(),
                );

                cur_cnr_prob.assign(&cur_c_prob);
                //cur_cnr_prob_e.assign(&cur_c_prob_e);

            } else {
                let mut prev_block_prob = prev_block_prob.unwrap();
                let prev_block = &blocks[block_i - 1];
                let rprob = rprobs_iter.next().unwrap();

                let (prev_alpha_pre, prev_alpha_post) = (
                    prev_block_prob.alpha_pre.view_mut(),
                    prev_block_prob.alpha_post.view_mut(),
                );

                let fprobs_save = Some(ForwardProbSave {
                    c: cur_block_prob.c_prob.slice_mut(s![0, .., ..]),
                    cnr: cur_block_prob.cnr_prob.slice_mut(s![0, .., ..]),
                    #[cfg(feature = "obliv")]
                    e: cur_block_prob.prob_e.slice_mut(s![0, ..]),
                });

                let do_collapse = genotype_graph[site_i].is_segment_marker();

                Self::block_transition(
                    prev_c_prob.view(),
                    #[cfg(feature = "obliv")]
                    prev_c_prob_e.view(),
                    prev_cnr_prob.view(),
                    #[cfg(feature = "obliv")]
                    prev_cnr_prob_e.view(),
                    prev_block,
                    block,
                    genotype_graph[site_i],
                    do_collapse,
                    #[cfg(not(feature = "obliv"))]
                    eprob,
                    rprob,
                    prev_alpha_pre.view(),
                    prev_alpha_post,
                    is_first_segment,
                    full_trans_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    full_trans_prob_e.view_mut(),
                    cur_c_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    cur_c_prob_e.view_mut(),
                    cur_cnr_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    cur_cnr_prob_e.view_mut(),
                    cur_block_prob.alpha_pre.view_mut(),
                    cur_block_prob.alpha_post.view_mut(),
                    fprobs_save,
                    cur_block_prob.save_alpha_post_cnr.view_mut(),
                );

                #[cfg(feature = "obliv")]
                {
                    is_first_segment = !do_collapse.expose();
                }

                #[cfg(not(feature = "obliv"))]
                {
                    is_first_segment = !do_collapse;
                }
            }

            cur_block_prob.is_pre[0] = true;

            site_i += 1;
            prev_c_prob = cur_c_prob;
            prev_cnr_prob = cur_cnr_prob;
            #[cfg(feature = "obliv")]
            {
                prev_c_prob_e = cur_c_prob_e;
                prev_cnr_prob_e = cur_cnr_prob_e;
            }

            for block_site_i in 1..block_n_sites {
                cur_c_prob = Array2::<Real>::zeros((P, block_n_unique_haps));
                cur_cnr_prob = Array2::<Real>::zeros((P, block_n_unique_haps));
                #[cfg(feature = "obliv")]
                {
                    cur_c_prob_e = Array1::from_elem(P, TpI16::protect(0));
                    cur_cnr_prob_e = Array1::from_elem(P, TpI16::protect(0));
                }

                let rprob = rprobs_iter.next().unwrap();

                let fprobs_save = Some(ForwardProbSave {
                    c: cur_block_prob.c_prob.slice_mut(s![block_site_i, .., ..]),
                    cnr: cur_block_prob.cnr_prob.slice_mut(s![block_site_i, .., ..]),
                    #[cfg(feature = "obliv")]
                    e: cur_block_prob.prob_e.row_mut(block_site_i),
                });
                cur_block_prob.is_pre[block_site_i] = is_first_segment;

                #[cfg(feature = "obliv")]
                let do_collapse = genotype_graph[site_i].is_segment_marker().expose();

                #[cfg(not(feature = "obliv"))]
                let do_collapse = genotype_graph[site_i].is_segment_marker();

                Self::nonblock_transition(
                    prev_c_prob.view(),
                    #[cfg(feature = "obliv")]
                    prev_c_prob_e.view(),
                    prev_cnr_prob.view(),
                    #[cfg(feature = "obliv")]
                    prev_cnr_prob_e.view(),
                    block,
                    genotype_graph[site_i],
                    do_collapse,
                    #[cfg(not(feature = "obliv"))]
                    eprob,
                    rprob,
                    cur_block_prob.alpha_pre.view(),
                    block_site_i,
                    is_first_segment,
                    cur_c_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    cur_c_prob_e.view_mut(),
                    cur_cnr_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    cur_cnr_prob_e.view_mut(),
                    cur_block_prob.alpha_post.view_mut(),
                    fprobs_save,
                    cur_block_prob.save_alpha_post_cnr.view_mut(),
                );

                if do_collapse {
                    is_first_segment = false;
                }

                site_i += 1;
                prev_c_prob = cur_c_prob;
                prev_cnr_prob = cur_cnr_prob;
                #[cfg(feature = "obliv")]
                {
                    prev_c_prob_e = cur_c_prob_e;
                    prev_cnr_prob_e = cur_cnr_prob_e;
                }
            }

            Self::combine_block(
                block,
                &cur_block_prob,
                &bprobs[block_i],
                tprobs.slice_mut(s![site_i - block_n_sites..site_i, .., ..]),
                #[cfg(feature = "obliv")]
                tprobs_e.slice_mut(s![site_i - block_n_sites..site_i, .., ..]),
            );

            prev_block_prob = Some(cur_block_prob);
        }

        #[cfg(feature = "obliv")]
        return (first_tprobs, first_tprobs_e, tprobs, tprobs_e);

        #[cfg(not(feature = "obliv"))]
        return (first_tprobs, tprobs);
    }

    pub fn backward<'a>(
        blocks: &[FilteredBlockSliceObliv<'a>],
        n_sites_window: usize,
        n_full_states: UInt,
        n_full_haps: usize,
        genotype_graph: ArrayView1<G>,
        eprob: Real,
        rprobs: &RprobsSlice,
        _ignored_sites: ArrayView1<Bool>,
    ) -> Vec<ProbBlock> {
        let n_blocks = blocks.len();

        #[cfg(feature = "obliv")]
        let mut rprobs_iter = rprobs.get_backward(n_full_states.as_u64());

        #[cfg(not(feature = "obliv"))]
        let mut rprobs_iter = rprobs.get_backward(n_full_states as usize);

        let mut site_i = n_sites_window - 1;
        let mut is_first_segment = true;

        let mut full_trans_prob = Array2::zeros((n_full_haps, P));

        #[cfg(feature = "obliv")]
        let mut full_trans_prob_e = Array1::from_elem(P, TpI16::protect(0));

        let mut all_prob_blocks = Vec::<ProbBlock>::new();

        for (block_i, block) in blocks.iter().enumerate().rev() {
            let block_n_sites = block.n_sites();
            let block_n_unique_haps = block.n_unique_haps();

            let mut cur_block_prob =
                ProbBlock::new(block_n_sites, block_n_unique_haps, n_full_haps);

            {
                if block_i == n_blocks - 1 {
                    let mut cur_c_prob = cur_block_prob
                        .c_prob
                        .slice_mut(s![block_n_sites - 1, .., ..]);

                    #[cfg(feature = "obliv")]
                    let cur_c_prob_e = cur_block_prob.prob_e.slice_mut(s![block_n_sites - 1, ..]);

                    let mut cur_cnr_prob =
                        cur_block_prob
                            .cnr_prob
                            .slice_mut(s![block_n_sites - 1, .., ..]);

                    Self::init(
                        false,
                        block,
                        genotype_graph[site_i],
                        eprob,
                        cur_c_prob.view_mut(),
                        #[cfg(feature = "obliv")]
                        cur_c_prob_e,
                        cur_block_prob.alpha_pre.view_mut(),
                    );

                    cur_cnr_prob.assign(&cur_c_prob);
                } else {
                    let prev_block_prob = all_prob_blocks.last_mut().unwrap();
                    let prev_block = &blocks[block_i + 1];

                    let rprob = rprobs_iter.next().unwrap();

                    let do_collapse = genotype_graph[site_i + 1].is_segment_marker();

                    Self::block_transition_backward(
                        prev_block_prob,
                        &mut cur_block_prob,
                        prev_block,
                        block,
                        genotype_graph[site_i],
                        do_collapse,
                        #[cfg(not(feature = "obliv"))]
                        eprob,
                        rprob,
                        is_first_segment,
                        full_trans_prob.view_mut(),
                        #[cfg(feature = "obliv")]
                        full_trans_prob_e.view_mut(),
                    );
                }

                cur_block_prob.is_pre[block_n_sites - 1] = true;

                if site_i > 0 {
                    site_i -= 1;
                }
            }

            is_first_segment = true;

            for block_site_i in (0..block_n_sites - 1).rev() {
                let (prev_c_prob, cur_c_prob) = cur_block_prob
                    .c_prob
                    .multi_slice_mut((s![block_site_i + 1, .., ..], s![block_site_i, .., ..]));

                let (prev_cnr_prob, mut cur_cnr_prob) = cur_block_prob
                    .cnr_prob
                    .multi_slice_mut((s![block_site_i + 1, .., ..], s![block_site_i, .., ..]));

                #[cfg(feature = "obliv")]
                let prev_c_prob_e = cur_block_prob.prob_e.row(block_site_i + 1);

                #[cfg(feature = "obliv")]
                let prev_cnr_prob_e = prev_c_prob_e.clone();

                #[cfg(feature = "obliv")]
                let mut cur_c_prob_e = Array1::from_elem(P, TpI16::protect(0));

                #[cfg(feature = "obliv")]
                let mut cur_cnr_prob_e = Array1::from_elem(P, TpI16::protect(0));

                let rprob = rprobs_iter.next().unwrap();

                #[cfg(feature = "obliv")]
                let do_collapse = genotype_graph[site_i + 1].is_segment_marker().expose();

                #[cfg(not(feature = "obliv"))]
                let do_collapse = genotype_graph[site_i + 1].is_segment_marker();

                Self::nonblock_transition(
                    prev_c_prob.view(),
                    #[cfg(feature = "obliv")]
                    prev_c_prob_e.view(),
                    prev_cnr_prob.view(),
                    #[cfg(feature = "obliv")]
                    prev_cnr_prob_e.view(),
                    block,
                    genotype_graph[site_i],
                    do_collapse,
                    #[cfg(not(feature = "obliv"))]
                    eprob,
                    rprob,
                    cur_block_prob.alpha_pre.view(),
                    block_site_i,
                    is_first_segment,
                    cur_c_prob,
                    #[cfg(feature = "obliv")]
                    cur_c_prob_e.view_mut(),
                    cur_cnr_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    cur_cnr_prob_e.view_mut(),
                    cur_block_prob.alpha_post.view_mut(),
                    None,
                    cur_block_prob.save_alpha_post_cnr.view_mut(),
                );

                if do_collapse {
                    is_first_segment = false;
                }

                cur_block_prob.is_pre[block_site_i] = is_first_segment;

                #[cfg(feature = "obliv")]
                {
                    Zip::from(&cur_c_prob_e)
                        .and(cur_cnr_prob.rows_mut())
                        .and(&mut cur_cnr_prob_e)
                        .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));
                    cur_block_prob
                        .prob_e
                        .slice_mut(s![block_site_i, ..])
                        .assign(&cur_c_prob_e);
                }

                if site_i > 0 {
                    site_i -= 1;
                }
            }
            all_prob_blocks.push(cur_block_prob);
        }
        all_prob_blocks.reverse();

        all_prob_blocks
    }

    fn init<'a>(
        is_forward: bool,
        block: &FilteredBlockSliceObliv<'a>,
        graph_pos: G,
        eprob: Real,
        mut c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut c_prob_e: ArrayViewMut1<TpI16>,
        mut alpha_pre: ArrayViewMut2<Real>,
    ) {
        let weights = block.weights.view();

        let inv_weights = block.inv_weights.view();

        let index_map = block.index_map.view();
        let weighted_eprobs = &weights * eprob;
        let block_site = if is_forward { 0 } else { block.n_sites() - 1 };

        Zip::indexed(c_prob.rows_mut()).for_each(|i, mut p_row| {
            Zip::from(&mut p_row)
                .and(&block.expand_pos(block_site))
                .and(&weighted_eprobs)
                .and(&weights)
                .for_each(|p, &z, &e, &w| {
                    #[cfg(feature = "obliv")]
                    {
                        let g = graph_pos.get_row(i);
                        *p = (z.tp_not_eq(&g)).select(e, w);
                    }

                    #[cfg(not(feature = "obliv"))]
                    if z != graph_pos.get_row(i) {
                        *p = e;
                    } else {
                        *p = w;
                    }
                });
        });

        #[cfg(feature = "obliv")]
        renorm_scale(c_prob.view_mut(), c_prob_e.view_mut());

        Zip::from(alpha_pre.rows_mut())
            .and(&index_map)
            .and(&block.full_filter)
            .for_each(|mut a, &i, &b| {
                #[cfg(feature = "obliv")]
                {
                    a.fill(b.select(inv_weights[i as usize], Real::ZERO))
                }

                #[cfg(not(feature = "obliv"))]
                if b {
                    a.fill(inv_weights[i as usize]);
                } else {
                    a.fill(0.);
                }
            });
    }

    fn nonblock_transition<'a, 'b>(
        prev_c_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_c_prob_e: ArrayView1<TpI16>,
        prev_cnr_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_cnr_prob_e: ArrayView1<TpI16>,
        block: &FilteredBlockSliceObliv<'a>,
        graph_pos: G,
        do_collapse: bool,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        rprob: Real,
        alpha_pre: ArrayView2<Real>,
        block_site_i: usize,
        is_first_segment: bool,
        mut cur_c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_c_prob_e: ArrayViewMut1<TpI16>,
        mut cur_cnr_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_cnr_prob_e: ArrayViewMut1<TpI16>,
        mut alpha_post: ArrayViewMut1<Real>,
        fprob_save: Option<ForwardProbSave<'b>>,
        save_cnr_prob: ArrayViewMut2<Real>,
    ) {
        let index_map = block.index_map.view();
        let weights = block.weights.view();

        let rprobs = &weights * rprob;

        Self::transition(
            rprobs.view(),
            prev_c_prob.view(),
            #[cfg(feature = "obliv")]
            prev_c_prob_e.view(),
            prev_cnr_prob.view(),
            #[cfg(feature = "obliv")]
            prev_cnr_prob_e.view(),
            cur_c_prob.view_mut(),
            #[cfg(feature = "obliv")]
            cur_c_prob_e.view_mut(),
            cur_cnr_prob.view_mut(),
            #[cfg(feature = "obliv")]
            cur_cnr_prob_e.view_mut(),
        );

        if let Some(mut fprob_save) = fprob_save {
            #[cfg(feature = "obliv")]
            {
                Zip::from(&cur_c_prob_e)
                    .and(cur_cnr_prob.rows_mut())
                    .and(&mut cur_cnr_prob_e)
                    .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));
                fprob_save.e.assign(&cur_cnr_prob_e);
            }
            fprob_save.c.assign(&cur_c_prob);
            fprob_save.cnr.assign(&cur_cnr_prob);
        }

        if do_collapse {
            Self::collapse(
                index_map.view(),
                is_first_segment,
                alpha_pre.view(),
                alpha_post.view_mut(),
                cur_c_prob.view_mut(),
                #[cfg(feature = "obliv")]
                cur_c_prob_e.view_mut(),
                cur_cnr_prob.view_mut(),
                #[cfg(feature = "obliv")]
                cur_cnr_prob_e.view_mut(),
                save_cnr_prob,
            );
        }

        let cur_ref_panel_pos = block.expand_pos(block_site_i);

        Self::emission(
            cur_ref_panel_pos.view(),
            graph_pos,
            #[cfg(not(feature = "obliv"))]
            eprob,
            cur_c_prob.view_mut(),
            #[cfg(feature = "obliv")]
            cur_c_prob_e.view_mut(),
        );

        Self::emission(
            cur_ref_panel_pos.view(),
            graph_pos,
            #[cfg(not(feature = "obliv"))]
            eprob,
            cur_cnr_prob.view_mut(),
            #[cfg(feature = "obliv")]
            cur_cnr_prob_e.view_mut(),
        );
    }

    fn block_transition_backward<'a, 'b>(
        prev_prob_block: &mut ProbBlock,
        cur_prob_block: &mut ProbBlock,
        prev_block: &FilteredBlockSliceObliv<'a>,
        cur_block: &FilteredBlockSliceObliv<'a>,
        cur_graph_pos: G,
        do_collapse: Bool,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        rprob: Real,
        mut is_first_segment: bool,
        mut full_trans_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut full_trans_prob_e: ArrayViewMut1<TpI16>,
    ) {
        let prev_weights = prev_block.weights.view();
        let prev_inv_weights = prev_block.inv_weights.view();
        let prev_index_map = prev_block.index_map.view();
        let prev_full_filter = prev_block.full_filter.view();

        let prev_c_prob = prev_prob_block.c_prob.slice(s![0, .., ..]);
        let prev_cnr_prob = prev_prob_block.cnr_prob.slice(s![0, .., ..]);

        #[cfg(feature = "obliv")]
        let prev_c_prob_e = prev_prob_block.prob_e.row(0);
        #[cfg(feature = "obliv")]
        let prev_cnr_prob_e = prev_prob_block.prob_e.row(0);

        let prev_alpha_pre = prev_prob_block.alpha_pre.view();
        let mut prev_alpha_post = prev_prob_block.alpha_post.view_mut();

        let cur_block_n_sites = cur_prob_block.n_sites();

        let mut cur_c_prob = cur_prob_block
            .c_prob
            .slice_mut(s![cur_block_n_sites - 1, .., ..]);
        let mut cur_cnr_prob = cur_prob_block
            .cnr_prob
            .slice_mut(s![cur_block_n_sites - 1, .., ..]);

        #[cfg(feature = "obliv")]
        let mut cur_c_prob_e = cur_prob_block.prob_e.row_mut(cur_block_n_sites - 1);

        let mut cur_alpha_pre = cur_prob_block.alpha_pre.view_mut();
        let cur_index_map = cur_block.index_map.view();
        let rprobs = &prev_weights * rprob;

        let mut trans_c_prob = Array2::<Real>::zeros(prev_c_prob.raw_dim());
        let mut trans_cnr_prob = Array2::<Real>::zeros(prev_cnr_prob.raw_dim());

        #[cfg(feature = "obliv")]
        let mut trans_c_prob_e = Array1::<TpI16>::from_elem(prev_c_prob.nrows(), TpI16::protect(0));
        #[cfg(feature = "obliv")]
        let mut trans_cnr_prob_e =
            Array1::<TpI16>::from_elem(prev_cnr_prob.nrows(), TpI16::protect(0));

        Self::transition(
            rprobs.view(),
            prev_c_prob.view(),
            #[cfg(feature = "obliv")]
            prev_c_prob_e.view(),
            prev_cnr_prob.view(),
            #[cfg(feature = "obliv")]
            prev_cnr_prob_e.view(),
            trans_c_prob.view_mut(),
            #[cfg(feature = "obliv")]
            trans_c_prob_e.view_mut(),
            trans_cnr_prob.view_mut(),
            #[cfg(feature = "obliv")]
            trans_cnr_prob_e.view_mut(),
        );

        #[cfg(feature = "obliv")]
        if do_collapse.expose() {
            Self::collapse(
                prev_index_map,
                is_first_segment,
                prev_alpha_pre.view(),
                prev_alpha_post.view_mut(),
                trans_c_prob.view_mut(),
                trans_c_prob_e.view_mut(),
                trans_cnr_prob.view_mut(),
                trans_cnr_prob_e.view_mut(),
                prev_prob_block.save_alpha_post_cnr.view_mut(),
            );

            is_first_segment = false;
        }

        #[cfg(not(feature = "obliv"))]
        if do_collapse {
            Self::collapse(
                prev_index_map,
                is_first_segment,
                prev_alpha_pre.view(),
                prev_alpha_post.view_mut(),
                trans_c_prob.view_mut(),
                trans_cnr_prob.view_mut(),
            );

            is_first_segment = false;
        }

        let t = Instant::now();
        #[cfg(feature = "obliv")]
        Zip::from(&trans_c_prob_e)
            .and(trans_cnr_prob.rows_mut())
            .and(&mut trans_cnr_prob_e)
            .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));

        let trans_cr_prob = &trans_c_prob - &trans_cnr_prob;

        #[cfg(feature = "obliv")]
        let trans_cr_prob_e = trans_c_prob_e.clone();

        BLOCK.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        Self::expand_prob(
            prev_index_map,
            prev_full_filter,
            prev_inv_weights.view(),
            trans_cr_prob.view(),
            #[cfg(feature = "obliv")]
            trans_cr_prob_e.view(),
            trans_cnr_prob.view(),
            #[cfg(feature = "obliv")]
            trans_cnr_prob_e.view(),
            prev_alpha_pre.view(),
            prev_alpha_post.view(),
            is_first_segment,
            full_trans_prob.view_mut(),
            #[cfg(feature = "obliv")]
            full_trans_prob_e.view_mut(),
        );

        let t = Instant::now();
        let mut cur_c_prob_ = Array::from_shape_vec(
            cur_c_prob.t().raw_dim(),
            cur_c_prob.t().iter().cloned().collect(),
        )
        .unwrap();
        Zip::from(full_trans_prob.rows())
            .and(&cur_index_map)
            .for_each(|p, &i| {
                let mut c = cur_c_prob_.row_mut(i as usize);
                c += &p;
            });

        #[cfg(feature = "obliv")]
        let (div_cur_c_prob_, div_cur_c_prob_e) = {
            let mut div_cur_c_prob_e =
                Array2::<TpI16>::from_elem(cur_c_prob_.raw_dim(), TpI16::protect(0));
            let mut div_cur_c_prob_ = cur_c_prob_.clone();
            Zip::from(&mut div_cur_c_prob_)
                .and(&mut div_cur_c_prob_e)
                .for_each(|p, e| renorm_scale_single(p, e));
            div_cur_c_prob_.map_mut(|v| {
                *v = v
                    .tp_eq(&Real::ZERO)
                    .select(Real::ZERO, Real::protect_i64(1) / *v)
            });
            (div_cur_c_prob_, div_cur_c_prob_e)
        };

        #[cfg(not(feature = "obliv"))]
        let div_cur_c_prob_ = cur_c_prob_.map(|&v| if v == 0. { 0. } else { 1. / v });

        Zip::from(cur_alpha_pre.rows_mut())
            .and(full_trans_prob.rows())
            .and(&cur_index_map)
            .for_each(|mut a, l, &i| {
                a.assign(&(&l * &div_cur_c_prob_.row(i as usize)));
                #[cfg(feature = "obliv")]
                {
                    let mut _e = TpI16::protect(0);
                    Zip::from(&mut a)
                        .and(&div_cur_c_prob_e.row(i as usize))
                        .for_each(|p, &e| adjust_scale_single(e, p, &mut _e));
                }
            });

        cur_c_prob.assign(&cur_c_prob_.t());

        #[cfg(feature = "obliv")]
        cur_c_prob_e.assign(&trans_c_prob_e);

        #[cfg(feature = "obliv")]
        renorm_scale(cur_c_prob.view_mut(), cur_c_prob_e.view_mut());

        BLOCK.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        Self::emission(
            cur_block.expand_pos(cur_block.n_sites() - 1).view(),
            cur_graph_pos,
            #[cfg(not(feature = "obliv"))]
            eprob,
            cur_c_prob.view_mut(),
            #[cfg(feature = "obliv")]
            cur_c_prob_e.view_mut(),
        );

        cur_cnr_prob.assign(&cur_c_prob);
    }

    fn block_transition_forward<'a, 'b>(
        prev_c_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_c_prob_e: ArrayView1<TpI16>,
        prev_cnr_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_cnr_prob_e: ArrayView1<TpI16>,
        prev_block: &FilteredBlockSliceObliv<'a>,
        cur_block: &FilteredBlockSliceObliv<'a>,
        cur_graph_pos: G,
        do_collapse: Bool,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        rprob: Real,
        prev_alpha_pre: ArrayView2<Real>,
        prev_alpha_post: ArrayViewMut1<Real>,
        is_first_segment: bool,
        mut full_trans_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut full_trans_prob_e: ArrayViewMut1<TpI16>,
        mut cur_c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_c_prob_e: ArrayViewMut1<TpI16>,
        mut cur_cnr_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_cnr_prob_e: ArrayViewMut1<TpI16>,
        mut cur_alpha_pre: ArrayViewMut2<Real>,
        mut cur_alpha_post: ArrayViewMut1<Real>,
        mut fprob_save: ForwardProbSave<'b>,
        save_cnr_prob: ArrayViewMut2<Real>,
    ) {
        let prev_weights = prev_block.weights.view();

        let prev_inv_weights = prev_block.inv_weights.view();

        let prev_index_map = prev_block.index_map.view();
        let prev_full_filter = prev_block.full_filter.view();
        let cur_index_map = cur_block.index_map.view();

        let rprobs = &prev_weights * rprob;

        let mut trans_c_prob = Array2::<Real>::zeros(prev_c_prob.raw_dim());

        #[cfg(feature = "obliv")]
        let mut trans_c_prob_e = Array1::<TpI16>::from_elem(prev_c_prob.nrows(), TpI16::protect(0));

        let mut trans_cnr_prob = Array2::<Real>::zeros(prev_cnr_prob.raw_dim());

        #[cfg(feature = "obliv")]
        let mut trans_cnr_prob_e =
            Array1::<TpI16>::from_elem(prev_cnr_prob.nrows(), TpI16::protect(0));

        Self::transition(
            rprobs.view(),
            prev_c_prob.view(),
            #[cfg(feature = "obliv")]
            prev_c_prob_e.view(),
            prev_cnr_prob.view(),
            #[cfg(feature = "obliv")]
            prev_cnr_prob_e.view(),
            trans_c_prob.view_mut(),
            #[cfg(feature = "obliv")]
            trans_c_prob_e.view_mut(),
            trans_cnr_prob.view_mut(),
            #[cfg(feature = "obliv")]
            trans_cnr_prob_e.view_mut(),
        );

        let t = Instant::now();
        #[cfg(feature = "obliv")]
        Zip::from(&trans_c_prob_e)
            .and(trans_cnr_prob.rows_mut())
            .and(&mut trans_cnr_prob_e)
            .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));

        let trans_cr_prob = &trans_c_prob - &trans_cnr_prob;

        #[cfg(feature = "obliv")]
        let trans_cr_prob_e = trans_c_prob_e.clone();

        BLOCK.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        Self::expand_prob(
            prev_index_map,
            prev_full_filter,
            prev_inv_weights.view(),
            trans_cr_prob.view(),
            #[cfg(feature = "obliv")]
            trans_cr_prob_e.view(),
            trans_cnr_prob.view(),
            #[cfg(feature = "obliv")]
            trans_cnr_prob_e.view(),
            prev_alpha_pre.view(),
            prev_alpha_post.view(),
            is_first_segment,
            full_trans_prob.view_mut(),
            #[cfg(feature = "obliv")]
            full_trans_prob_e.view_mut(),
        );

        let t = Instant::now();
        let mut cur_c_prob_ = Array::from_shape_vec(
            cur_c_prob.t().raw_dim(),
            cur_c_prob.t().iter().cloned().collect(),
        )
        .unwrap();

        Zip::from(full_trans_prob.rows())
            .and(&cur_index_map)
            .for_each(|p, &i| {
                let mut c = cur_c_prob_.row_mut(i as usize);
                c += &p;
            });

        cur_c_prob.assign(&cur_c_prob_.t());

        #[cfg(feature = "obliv")]
        cur_c_prob_e.assign(&trans_c_prob_e);

        #[cfg(feature = "obliv")]
        renorm_scale(cur_c_prob.view_mut(), cur_c_prob_e.view_mut());

        #[cfg(feature = "obliv")]
        {
            Zip::from(&cur_c_prob_e)
                .and(cur_cnr_prob.rows_mut())
                .and(&mut cur_cnr_prob_e)
                .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));
            fprob_save.e.assign(&cur_cnr_prob_e);
        }
        fprob_save.c.assign(&cur_c_prob);
        fprob_save.cnr.assign(&cur_c_prob);

        #[cfg(feature = "obliv")]
        let (div_cur_c_prob_, div_cur_c_prob_e) = {
            let mut div_cur_c_prob_e =
                Array2::<TpI16>::from_elem(cur_c_prob_.raw_dim(), TpI16::protect(0));
            Zip::from(&mut cur_c_prob_)
                .and(&mut div_cur_c_prob_e)
                .for_each(|p, e| renorm_scale_single(p, e));
            (
                cur_c_prob_.map(|&v| {
                    v.tp_eq(&Real::ZERO)
                        .select(Real::ZERO, Real::protect_i64(1) / v)
                }),
                div_cur_c_prob_e,
            )
        };

        #[cfg(not(feature = "obliv"))]
        let div_cur_c_prob_ = cur_c_prob_.map(|&v| if v == 0. { 0. } else { 1. / v });

        Zip::from(cur_alpha_pre.rows_mut())
            .and(full_trans_prob.rows())
            .and(&cur_index_map)
            .for_each(|mut a, l, &i| {
                a.assign(&(&l * &div_cur_c_prob_.row(i as usize)));
                #[cfg(feature = "obliv")]
                {
                    let mut _e = TpI16::protect(0);
                    Zip::from(&mut a)
                        .and(&div_cur_c_prob_e.row(i as usize))
                        .for_each(|p, &e| adjust_scale_single(e, p, &mut _e));
                }
            });

        cur_cnr_prob.assign(&cur_c_prob);

        #[cfg(feature = "obliv")]
        cur_cnr_prob_e.assign(&cur_c_prob_e);

        BLOCK.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        #[cfg(feature = "obliv")]
        if do_collapse.expose() {
            Self::collapse(
                cur_index_map,
                true,
                cur_alpha_pre.view(),
                cur_alpha_post.view_mut(),
                cur_c_prob.view_mut(),
                cur_c_prob_e.view_mut(),
                cur_cnr_prob.view_mut(),
                cur_cnr_prob_e.view_mut(),
                save_cnr_prob,
            );
        }

        #[cfg(not(feature = "obliv"))]
        if do_collapse {
            Self::collapse(
                cur_index_map,
                true,
                cur_alpha_pre.view(),
                cur_alpha_post.view_mut(),
                cur_c_prob.view_mut(),
                cur_cnr_prob.view_mut(),
            );
        }
        let cur_block_site = 0;
        Self::emission(
            cur_block.expand_pos(cur_block_site).view(),
            cur_graph_pos,
            #[cfg(not(feature = "obliv"))]
            eprob,
            cur_c_prob.view_mut(),
            #[cfg(feature = "obliv")]
            cur_c_prob_e.view_mut(),
        );
        Self::emission(
            cur_block.expand_pos(cur_block_site).view(),
            cur_graph_pos,
            #[cfg(not(feature = "obliv"))]
            eprob,
            cur_cnr_prob.view_mut(),
            #[cfg(feature = "obliv")]
            cur_cnr_prob_e.view_mut(),
        );
    }

    fn block_transition<'a, 'b>(
        prev_c_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_c_prob_e: ArrayView1<TpI16>,
        prev_cnr_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_cnr_prob_e: ArrayView1<TpI16>,
        prev_block: &FilteredBlockSliceObliv<'a>,
        cur_block: &FilteredBlockSliceObliv<'a>,
        cur_graph_pos: G,
        do_collapse: Bool,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        rprob: Real,
        prev_alpha_pre: ArrayView2<Real>,
        mut prev_alpha_post: ArrayViewMut1<Real>,
        mut is_first_segment: bool,
        mut full_trans_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut full_trans_prob_e: ArrayViewMut1<TpI16>,
        mut cur_c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_c_prob_e: ArrayViewMut1<TpI16>,
        mut cur_cnr_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_cnr_prob_e: ArrayViewMut1<TpI16>,
        mut cur_alpha_pre: ArrayViewMut2<Real>,
        mut cur_alpha_post: ArrayViewMut1<Real>,
        fprob_save: Option<ForwardProbSave<'b>>,
        save_cnr_prob: ArrayViewMut2<Real>,
    ) {
        let prev_weights = prev_block.weights.view();

        let prev_inv_weights = prev_block.inv_weights.view();

        let prev_index_map = prev_block.index_map.view();
        let prev_full_filter = prev_block.full_filter.view();
        let cur_index_map = cur_block.index_map.view();

        let rprobs = &prev_weights * rprob;

        let mut trans_c_prob = Array2::<Real>::zeros(prev_c_prob.raw_dim());

        #[cfg(feature = "obliv")]
        let mut trans_c_prob_e = Array1::<TpI16>::from_elem(prev_c_prob.nrows(), TpI16::protect(0));

        let mut trans_cnr_prob = Array2::<Real>::zeros(prev_cnr_prob.raw_dim());

        #[cfg(feature = "obliv")]
        let mut trans_cnr_prob_e =
            Array1::<TpI16>::from_elem(prev_cnr_prob.nrows(), TpI16::protect(0));

        Self::transition(
            rprobs.view(),
            prev_c_prob.view(),
            #[cfg(feature = "obliv")]
            prev_c_prob_e.view(),
            prev_cnr_prob.view(),
            #[cfg(feature = "obliv")]
            prev_cnr_prob_e.view(),
            trans_c_prob.view_mut(),
            #[cfg(feature = "obliv")]
            trans_c_prob_e.view_mut(),
            trans_cnr_prob.view_mut(),
            #[cfg(feature = "obliv")]
            trans_cnr_prob_e.view_mut(),
        );

        if let Some(mut fprob_save) = fprob_save {
            let t = Instant::now();
            #[cfg(feature = "obliv")]
            Zip::from(&trans_c_prob_e)
                .and(trans_cnr_prob.rows_mut())
                .and(&mut trans_cnr_prob_e)
                .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));

            let trans_cr_prob = &trans_c_prob - &trans_cnr_prob;

            #[cfg(feature = "obliv")]
            let trans_cr_prob_e = trans_c_prob_e.clone();

            BLOCK.with(|v| {
                let mut v = v.borrow_mut();
                *v += t.elapsed();
            });

            Self::expand_prob(
                prev_index_map,
                prev_full_filter,
                prev_inv_weights.view(),
                trans_cr_prob.view(),
                #[cfg(feature = "obliv")]
                trans_cr_prob_e.view(),
                trans_cnr_prob.view(),
                #[cfg(feature = "obliv")]
                trans_cnr_prob_e.view(),
                prev_alpha_pre.view(),
                prev_alpha_post.view(),
                is_first_segment,
                full_trans_prob.view_mut(),
                #[cfg(feature = "obliv")]
                full_trans_prob_e.view_mut(),
            );

            let t = Instant::now();
            let mut cur_c_prob_ = Array::from_shape_vec(
                cur_c_prob.t().raw_dim(),
                cur_c_prob.t().iter().cloned().collect(),
            )
            .unwrap();

            Zip::from(full_trans_prob.rows())
                .and(&cur_index_map)
                .for_each(|p, &i| {
                    let mut c = cur_c_prob_.row_mut(i as usize);
                    c += &p;
                });

            cur_c_prob.assign(&cur_c_prob_.t());

            #[cfg(feature = "obliv")]
            cur_c_prob_e.assign(&trans_c_prob_e);

            #[cfg(feature = "obliv")]
            renorm_scale(cur_c_prob.view_mut(), cur_c_prob_e.view_mut());

            #[cfg(feature = "obliv")]
            {
                Zip::from(&cur_c_prob_e)
                    .and(cur_cnr_prob.rows_mut())
                    .and(&mut cur_cnr_prob_e)
                    .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));
                fprob_save.e.assign(&cur_cnr_prob_e);
            }
            fprob_save.c.assign(&cur_c_prob);
            fprob_save.cnr.assign(&cur_c_prob);

            #[cfg(feature = "obliv")]
            let (div_cur_c_prob_, div_cur_c_prob_e) = {
                let mut div_cur_c_prob_e =
                    Array2::<TpI16>::from_elem(cur_c_prob_.raw_dim(), TpI16::protect(0));
                Zip::from(&mut cur_c_prob_)
                    .and(&mut div_cur_c_prob_e)
                    .for_each(|p, e| renorm_scale_single(p, e));
                (
                    cur_c_prob_.map(|&v| {
                        v.tp_eq(&Real::ZERO)
                            .select(Real::ZERO, Real::protect_i64(1) / v)
                    }),
                    div_cur_c_prob_e,
                )
            };

            #[cfg(not(feature = "obliv"))]
            let div_cur_c_prob_ = cur_c_prob_.map(|&v| if v == 0. { 0. } else { 1. / v });

            Zip::from(cur_alpha_pre.rows_mut())
                .and(full_trans_prob.rows())
                .and(&cur_index_map)
                .for_each(|mut a, l, &i| {
                    a.assign(&(&l * &div_cur_c_prob_.row(i as usize)));
                    #[cfg(feature = "obliv")]
                    {
                        let mut _e = TpI16::protect(0);
                        Zip::from(&mut a)
                            .and(&div_cur_c_prob_e.row(i as usize))
                            .for_each(|p, &e| adjust_scale_single(e, p, &mut _e));
                    }
                });

            cur_cnr_prob.assign(&cur_c_prob);

            #[cfg(feature = "obliv")]
            cur_cnr_prob_e.assign(&cur_c_prob_e);

            BLOCK.with(|v| {
                let mut v = v.borrow_mut();
                *v += t.elapsed();
            });

            #[cfg(feature = "obliv")]
            if do_collapse.expose() {
                Self::collapse(
                    cur_index_map,
                    true,
                    cur_alpha_pre.view(),
                    cur_alpha_post.view_mut(),
                    cur_c_prob.view_mut(),
                    cur_c_prob_e.view_mut(),
                    cur_cnr_prob.view_mut(),
                    cur_cnr_prob_e.view_mut(),
                    save_cnr_prob,
                );
            }

            #[cfg(not(feature = "obliv"))]
            if do_collapse {
                Self::collapse(
                    cur_index_map,
                    true,
                    cur_alpha_pre.view(),
                    cur_alpha_post.view_mut(),
                    cur_c_prob.view_mut(),
                    cur_cnr_prob.view_mut(),
                );
            }
            let cur_block_site = 0;
            Self::emission(
                cur_block.expand_pos(cur_block_site).view(),
                cur_graph_pos,
                #[cfg(not(feature = "obliv"))]
                eprob,
                cur_c_prob.view_mut(),
                #[cfg(feature = "obliv")]
                cur_c_prob_e.view_mut(),
            );
            Self::emission(
                cur_block.expand_pos(cur_block_site).view(),
                cur_graph_pos,
                #[cfg(not(feature = "obliv"))]
                eprob,
                cur_cnr_prob.view_mut(),
                #[cfg(feature = "obliv")]
                cur_cnr_prob_e.view_mut(),
            );
        } else {
            #[cfg(feature = "obliv")]
            if do_collapse.expose() {
                Self::collapse(
                    prev_index_map,
                    is_first_segment,
                    prev_alpha_pre.view(),
                    prev_alpha_post.view_mut(),
                    trans_c_prob.view_mut(),
                    trans_c_prob_e.view_mut(),
                    trans_cnr_prob.view_mut(),
                    trans_cnr_prob_e.view_mut(),
                    save_cnr_prob,
                );

                is_first_segment = false;
            }

            #[cfg(not(feature = "obliv"))]
            if do_collapse {
                Self::collapse(
                    prev_index_map,
                    is_first_segment,
                    prev_alpha_pre.view(),
                    prev_alpha_post.view_mut(),
                    trans_c_prob.view_mut(),
                    trans_cnr_prob.view_mut(),
                );

                is_first_segment = false;
            }

            let t = Instant::now();
            #[cfg(feature = "obliv")]
            Zip::from(&trans_c_prob_e)
                .and(trans_cnr_prob.rows_mut())
                .and(&mut trans_cnr_prob_e)
                .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));

            let trans_cr_prob = &trans_c_prob - &trans_cnr_prob;

            #[cfg(feature = "obliv")]
            let trans_cr_prob_e = trans_c_prob_e.clone();

            BLOCK.with(|v| {
                let mut v = v.borrow_mut();
                *v += t.elapsed();
            });

            Self::expand_prob(
                prev_index_map,
                prev_full_filter,
                prev_inv_weights.view(),
                trans_cr_prob.view(),
                #[cfg(feature = "obliv")]
                trans_cr_prob_e.view(),
                trans_cnr_prob.view(),
                #[cfg(feature = "obliv")]
                trans_cnr_prob_e.view(),
                prev_alpha_pre.view(),
                prev_alpha_post.view(),
                is_first_segment,
                full_trans_prob.view_mut(),
                #[cfg(feature = "obliv")]
                full_trans_prob_e.view_mut(),
            );

            let t = Instant::now();
            let mut cur_c_prob_ = Array::from_shape_vec(
                cur_c_prob.t().raw_dim(),
                cur_c_prob.t().iter().cloned().collect(),
            )
            .unwrap();
            Zip::from(full_trans_prob.rows())
                .and(&cur_index_map)
                .for_each(|p, &i| {
                    let mut c = cur_c_prob_.row_mut(i as usize);
                    c += &p;
                });

            #[cfg(feature = "obliv")]
            let (div_cur_c_prob_, div_cur_c_prob_e) = {
                let mut div_cur_c_prob_e =
                    Array2::<TpI16>::from_elem(cur_c_prob_.raw_dim(), TpI16::protect(0));
                let mut div_cur_c_prob_ = cur_c_prob_.clone();
                Zip::from(&mut div_cur_c_prob_)
                    .and(&mut div_cur_c_prob_e)
                    .for_each(|p, e| renorm_scale_single(p, e));
                div_cur_c_prob_.map_mut(|v| {
                    *v = v
                        .tp_eq(&Real::ZERO)
                        .select(Real::ZERO, Real::protect_i64(1) / *v)
                });
                (div_cur_c_prob_, div_cur_c_prob_e)
            };

            #[cfg(not(feature = "obliv"))]
            let div_cur_c_prob_ = cur_c_prob_.map(|&v| if v == 0. { 0. } else { 1. / v });

            Zip::from(cur_alpha_pre.rows_mut())
                .and(full_trans_prob.rows())
                .and(&cur_index_map)
                .for_each(|mut a, l, &i| {
                    a.assign(&(&l * &div_cur_c_prob_.row(i as usize)));
                    #[cfg(feature = "obliv")]
                    {
                        let mut _e = TpI16::protect(0);
                        Zip::from(&mut a)
                            .and(&div_cur_c_prob_e.row(i as usize))
                            .for_each(|p, &e| adjust_scale_single(e, p, &mut _e));
                    }
                });

            let cur_block_site = cur_block.n_sites() - 1;

            cur_c_prob.assign(&cur_c_prob_.t());

            #[cfg(feature = "obliv")]
            cur_c_prob_e.assign(&trans_c_prob_e);

            #[cfg(feature = "obliv")]
            renorm_scale(cur_c_prob.view_mut(), cur_c_prob_e.view_mut());

            BLOCK.with(|v| {
                let mut v = v.borrow_mut();
                *v += t.elapsed();
            });

            Self::emission(
                cur_block.expand_pos(cur_block_site).view(),
                cur_graph_pos,
                #[cfg(not(feature = "obliv"))]
                eprob,
                cur_c_prob.view_mut(),
                #[cfg(feature = "obliv")]
                cur_c_prob_e.view_mut(),
            );

            cur_cnr_prob.assign(&cur_c_prob);

            #[cfg(feature = "obliv")]
            cur_cnr_prob_e.assign(&cur_c_prob_e);
        }
    }

    fn emission(
        cond_haps: ArrayView1<i8>,
        graph_col: G,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        mut probs: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut probs_e: ArrayViewMut1<TpI16>,
    ) {
        let t = Instant::now();

        Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
            Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
                #[cfg(feature = "obliv")]
                {
                    let z = TpI8::protect(z);
                    let g = graph_col.get_row(i);
                    *p = (z.tp_not_eq(&g))
                        .select((*p >> 14) + (*p >> 15) + (*p >> 17) + (*p >> 20), *p);
                }

                #[cfg(not(feature = "obliv"))]
                if z != graph_col.get_row(i) {
                    *p *= eprob;
                }
            });
        });

        #[cfg(feature = "obliv")]
        renorm_scale(probs.view_mut(), probs_e.view_mut());

        EMISS.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }

    fn transition(
        rprobs: ArrayView1<Real>,
        prev_c_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_c_prob_e: ArrayView1<TpI16>,
        prev_cnr_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_cnr_prob_e: ArrayView1<TpI16>,
        mut cur_c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_c_prob_e: ArrayViewMut1<TpI16>,
        mut cur_cnr_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_cnr_prob_e: ArrayViewMut1<TpI16>,
    ) {
        let t = Instant::now();

        Zip::from(cur_c_prob.rows_mut())
            .and(prev_c_prob.rows())
            .for_each(|mut cr, pr| cr.assign(&(&pr + &rprobs * pr.sum())));

        cur_cnr_prob.assign(&prev_cnr_prob);

        #[cfg(feature = "obliv")]
        {
            cur_c_prob_e.assign(&prev_c_prob_e);
            cur_cnr_prob_e.assign(&prev_cnr_prob_e);
            renorm_e_pair(cur_c_prob_e.view_mut(), cur_cnr_prob_e.view_mut());
            renorm_scale(cur_c_prob.view_mut(), cur_c_prob_e.view_mut());
        }

        #[cfg(not(feature = "obliv"))]
        {
            let sum = cur_c_prob.sum();
            if sum < RENORM_THRESHOLD {
                cur_c_prob *= 1. / sum;
                cur_cnr_prob *= 1. / sum;
            }
        }

        TRANS.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }

    fn collapse(
        index_map: ArrayView1<u16>,
        do_compute_alpha_post: bool,
        alpha_pre: ArrayView2<Real>,
        alpha_post: ArrayViewMut1<Real>,
        mut cur_c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_c_prob_e: ArrayViewMut1<TpI16>,
        mut cur_cnr_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_cnr_prob_e: ArrayViewMut1<TpI16>,
        mut save_cnr_prob: ArrayViewMut2<Real>,
    ) {
        let t = Instant::now();

        #[cfg(feature = "obliv")]
        let (sum_c, sum_c_e) =
            sum_scale_by_column_mut(cur_c_prob.view_mut(), cur_c_prob_e.view_mut());

        #[cfg(not(feature = "obliv"))]
        let mut sum_c = Zip::from(cur_c_prob.columns()).map_collect(|c| c.sum());

        #[cfg(feature = "obliv")]
        let mut sum_cnr_e = max_e(cur_cnr_prob_e.view());

        #[cfg(feature = "obliv")]
        match_scale(
            sum_cnr_e,
            cur_cnr_prob.view_mut(),
            cur_cnr_prob_e.view_mut(),
        );

        let mut sum_cnr = Zip::from(cur_cnr_prob.columns()).map_collect(|c| c.sum());

        if do_compute_alpha_post {
            Self::compute_alpha_post(
                sum_cnr.view(),
                cur_cnr_prob.view(),
                alpha_pre,
                index_map,
                alpha_post,
            );

            save_cnr_prob.assign(&cur_cnr_prob);
        }

        #[cfg(not(feature = "obliv"))]
        {
            let sum = sum_c.sum();
            if sum < RENORM_THRESHOLD {
                sum_c *= 1. / sum;
                sum_cnr *= 1. / sum;
            }
        }
        #[cfg(feature = "obliv")]
        {
            renorm_scale_row(sum_cnr.view_mut(), &mut sum_cnr_e);
        }

        Zip::from(cur_c_prob.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_c));
        Zip::from(cur_cnr_prob.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_cnr));

        #[cfg(feature = "obliv")]
        {
            cur_c_prob_e.fill(sum_c_e);
            cur_cnr_prob_e.fill(sum_cnr_e);
        }

        #[cfg(feature = "obliv")]
        renorm_e_pair(cur_c_prob_e.view_mut(), cur_cnr_prob_e.view_mut());

        COLL.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }

    fn compute_alpha_post(
        sum_cnr: ArrayView1<Real>,
        cur_cnr_prob: ArrayView2<Real>,
        alpha_pre: ArrayView2<Real>,
        index_map: ArrayView1<u16>,
        mut alpha_post: ArrayViewMut1<Real>,
    ) {
        #[cfg(feature = "obliv")]
        let weighted_cur_cnr_prob = {
            let mut div_sum_cnr = sum_cnr.to_owned();
            let mut div_sum_cnr_e =
                Array1::<TpI16>::from_elem(div_sum_cnr.raw_dim(), TpI16::protect(0));
            Zip::from(&mut div_sum_cnr)
                .and(&mut div_sum_cnr_e)
                .for_each(|d, e| renorm_scale_single(d, e));
            div_sum_cnr.map_mut(|v| *v = Real::protect_i64(1) / *v);
            let mut weighted_cur_cnr_prob = cur_cnr_prob.to_owned();
            let mut _e = TpI16::protect(0);
            Zip::from(weighted_cur_cnr_prob.rows_mut()).for_each(|mut r| {
                r.assign(&(&r * &div_sum_cnr));
                Zip::from(&mut r).and(&div_sum_cnr_e).for_each(|p, &e| {
                    adjust_scale_single(e, p, &mut _e);
                    _e = TpI16::protect(0);
                });
            });
            weighted_cur_cnr_prob
        };

        //#[cfg(not(feature = "obliv"))]
        //let weighted_cur_cnr_prob = {
        //let div_sum_cnr = sum_cnr.map(|&v| 1. / v);
        //let mut weighted_cur_cnr_prob = cur_cnr_prob.to_owned();
        //Zip::from(weighted_cur_cnr_prob.rows_mut())
        //.for_each(|mut r| r.assign(&(&r * &div_sum_cnr)));
        //weighted_cur_cnr_prob
        //};

        Zip::from(&mut alpha_post)
            .and(alpha_pre.rows())
            .and(&index_map)
            .for_each(|a_post, a_pre, &i| {
                let i = i as usize;
                #[cfg(feature = "obliv")]
                {
                    *a_post = sum_cnr[[i]].tp_eq(&Real::ZERO).select(
                        Real::ZERO,
                        Dot::dot(&weighted_cur_cnr_prob.column(i), &a_pre),
                    );
                }

                #[cfg(not(feature = "obliv"))]
                {
                    *a_post = if sum_cnr[[i]] == 0. {
                        0.
                    } else {
                        cur_cnr_prob.column(i).dot(&a_pre) / sum_cnr[i]
                    };
                }
            });
    }

    fn expand_prob(
        index_map: ArrayView1<u16>,
        full_filter: ArrayView1<Bool>,
        inv_weights: ArrayView1<Real>,
        cr_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] cr_prob_e: ArrayView1<TpI16>,
        cnr_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] cnr_prob_e: ArrayView1<TpI16>,
        alpha_pre: ArrayView2<Real>,
        alpha_post: ArrayView1<Real>,
        is_pre: bool,
        mut expanded_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut expanded_prob_e: ArrayViewMut1<TpI16>,
    ) {
        let t = Instant::now();

        #[cfg(feature = "obliv")]
        let mut cr_prob = cr_prob.to_owned();
        #[cfg(feature = "obliv")]
        let mut cr_prob_e = cr_prob_e.to_owned();
        #[cfg(feature = "obliv")]
        let mut cnr_prob = cnr_prob.to_owned();
        #[cfg(feature = "obliv")]
        let mut cnr_prob_e = cnr_prob_e.to_owned();

        #[cfg(feature = "obliv")]
        Zip::from(&mut expanded_prob_e)
            .and(cr_prob.rows_mut())
            .and(&mut cr_prob_e)
            .and(cnr_prob.rows_mut())
            .and(&mut cnr_prob_e)
            .for_each(|expand_e, cr, cr_e, cnr, cnr_e| {
                *expand_e = cr_e.tp_gt(cnr_e).select(*cr_e, *cnr_e);
                match_scale_row(*expand_e, cr, cr_e);
                match_scale_row(*expand_e, cnr, cnr_e);
            });

        let mut cr_prob =
            Array::from_shape_vec(cr_prob.t().raw_dim(), cr_prob.t().iter().cloned().collect())
                .unwrap();

        Zip::from(cr_prob.rows_mut())
            .and(&inv_weights)
            .for_each(|mut p, &w| p *= w);

        let cnr_prob = Array::from_shape_vec(
            cnr_prob.t().raw_dim(),
            cnr_prob.t().iter().cloned().collect(),
        )
        .unwrap();

        if is_pre {
            Zip::from(expanded_prob.rows_mut())
                .and(alpha_pre.rows())
                .and(&index_map)
                .and(&full_filter)
                .for_each(|mut p, a, &i, &b| {
                    #[cfg(feature = "obliv")]
                    {
                        let i = i as usize;
                        let cr = cr_prob.row(i);
                        let cnr = cnr_prob.row(i);
                        let tmp = &cr + &cnr * &a;
                        Zip::from(&mut p).and(&tmp).for_each(|tar, &src| {
                            *tar = b.select(src, Real::ZERO);
                        });
                    }

                    #[cfg(not(feature = "obliv"))]
                    if b {
                        let i = i as usize;
                        let cr = cr_prob.row(i);
                        let cnr = cnr_prob.row(i);
                        p.assign(&(&cr + &cnr * &a));
                    } else {
                        p.fill(0.);
                    }
                });
        } else {
            Zip::from(expanded_prob.rows_mut())
                .and(&alpha_post)
                .and(&index_map)
                .and(&full_filter)
                .for_each(|mut p, &a, &i, &b| {
                    #[cfg(feature = "obliv")]
                    {
                        let i = i as usize;
                        let cr = cr_prob.row(i);
                        let cnr = cnr_prob.row(i);
                        let tmp = &cr + &cnr * a;
                        Zip::from(&mut p).and(&tmp).for_each(|tar, &src| {
                            *tar = b.select(src, Real::ZERO);
                        });
                    }
                    #[cfg(not(feature = "obliv"))]
                    if b {
                        let i = i as usize;
                        let cr = cr_prob.row(i);
                        let cnr = cnr_prob.row(i);
                        p.assign(&(&cr + &cnr * a));
                    } else {
                        p.fill(0.);
                    }
                });
        };

        EXPAND.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }

    fn combine_block<'a>(
        block: &FilteredBlockSliceObliv<'a>,
        forward: &ProbBlock,
        backward: &ProbBlock,
        mut tprobs: ArrayViewMut3<Real>,
        #[cfg(feature = "obliv")] mut tprobs_e: ArrayViewMut3<TpI16>,
    ) {
        {
            let forward_sum_cnr =
                Zip::from(forward.save_alpha_post_cnr.columns()).map_collect(|c| c.sum());
            let mut forward_test_alpha_post =
                Array1::from_elem(forward.alpha_post.dim(), Real::ZERO);
            Self::compute_alpha_post(
                forward_sum_cnr.view(),
                forward.save_alpha_post_cnr.view(),
                forward.alpha_pre.view(),
                block.index_map.view(),
                forward_test_alpha_post.view_mut(),
            );
            Zip::from(&forward.alpha_post)
                .and(&forward_test_alpha_post)
                .for_each(|a, b| assert_eq!(a.expose_into_f32(), b.expose_into_f32()));

            let backward_sum_cnr =
                Zip::from(backward.save_alpha_post_cnr.columns()).map_collect(|c| c.sum());
            let mut backward_test_alpha_post =
                Array1::from_elem(backward.alpha_post.dim(), Real::ZERO);
            Self::compute_alpha_post(
                backward_sum_cnr.view(),
                backward.save_alpha_post_cnr.view(),
                backward.alpha_pre.view(),
                block.index_map.view(),
                backward_test_alpha_post.view_mut(),
            );
            Zip::from(&backward.alpha_post)
                .and(&backward_test_alpha_post)
                .for_each(|a, b| assert_eq!(a.expose_into_f32(), b.expose_into_f32()));
        }

        let t = Instant::now();

        let inv_weights = block.inv_weights.view();

        let n_unique_haps = block.n_unique_haps();

        let mut alpha_11 = Array3::<Real>::zeros((P, P, n_unique_haps));
        let mut alpha_10 = Array2::<Real>::zeros((P, n_unique_haps));
        let mut alpha_01 = Array2::<Real>::zeros((P, n_unique_haps));
        let mut alpha_00 = Array1::<Real>::zeros(n_unique_haps);

        let mut arr = vec![Vec::new(); n_unique_haps];

        Zip::from(&block.index_map)
            .and(forward.alpha_pre.rows())
            .and(&forward.alpha_post)
            .and(backward.alpha_pre.rows())
            .and(&backward.alpha_post)
            .for_each(|&i, f_pre, &f_post, b_pre, &b_post| {
                arr[i as usize].push((f_pre, f_post, b_pre, b_post));
            });

        for (i, u) in arr.into_iter().enumerate() {
            let mut f_pre_arr = Array2::<Real>::zeros((P, u.len()));
            let mut f_post_arr = Array1::<Real>::zeros(u.len());
            let mut b_pre_arr = Array2::<Real>::zeros((P, u.len()));
            let mut b_post_arr = Array1::<Real>::zeros(u.len());

            for (j, (f_pre, f_post, b_pre, b_post)) in u.into_iter().enumerate() {
                f_pre_arr.column_mut(j).assign(&f_pre);
                f_post_arr[j] = f_post;
                b_pre_arr.column_mut(j).assign(&b_pre);
                b_post_arr[j] = b_post;
            }

            alpha_11
                .slice_mut(s![.., .., i])
                .assign(&Dot::dot(&f_pre_arr, &b_pre_arr.t()));
            alpha_10
                .slice_mut(s![.., i])
                .assign(&Dot::dot(&f_pre_arr, &b_post_arr));
            alpha_01
                .slice_mut(s![.., i])
                .assign(&Dot::dot(&b_pre_arr, &f_post_arr.t()));
            alpha_00[i] = Dot::dot(&f_post_arr, &b_post_arr);
        }

        COMB1.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        let t = Instant::now();

        Zip::indexed(tprobs.outer_iter_mut())
            .and(forward.c_prob.outer_iter())
            .and(forward.cnr_prob.outer_iter())
            .and(backward.c_prob.outer_iter())
            .and(backward.cnr_prob.outer_iter())
            .for_each(|i, mut t, fc, fcnr, bc, bcnr| {
                // first part
                let mut a = fc.to_owned();
                Zip::from(a.columns_mut())
                    .and(&inv_weights)
                    .for_each(|mut a_i, &r| a_i *= r);
                let mut b = fcnr.to_owned();
                Zip::from(b.columns_mut())
                    .and(&inv_weights)
                    .for_each(|mut b_i, &r| b_i *= r);
                let c = Dot::dot(&a, &bc.t()) - Dot::dot(&b, &bcnr.t());

                // second part
                if forward.is_pre[i] == false {
                    if backward.is_pre[i] == false {
                        let mut d = fcnr.to_owned();
                        Zip::from(d.columns_mut())
                            .and(&alpha_00)
                            .for_each(|mut e_i, &a| e_i *= a);
                        t.assign(&(c + Dot::dot(&d, &bcnr.t())));
                    } else {
                        let d = &bcnr * &alpha_01;
                        t.assign(&(c + Dot::dot(&fcnr, &d.t())));
                    }
                } else {
                    if backward.is_pre[i] == false {
                        let d = &fcnr * &alpha_10;
                        t.assign(&(c + Dot::dot(&d, &bcnr.t())));
                    } else {
                        Zip::from(alpha_11.outer_iter())
                            .and(fcnr.outer_iter())
                            .and(t.rows_mut())
                            .for_each(|a1, f, mut t_r| {
                                Zip::from(a1.outer_iter())
                                    .and(bcnr.outer_iter())
                                    .and(&mut t_r)
                                    .for_each(|a2, b, t_e| {
                                        *t_e = Dot::dot(&(&a2 * &f), &b.t());
                                    });
                            });
                        t += &c;
                    }
                }

                #[cfg(feature = "obliv")]
                Zip::from(tprobs_e.slice_mut(s![i, .., ..]).rows_mut())
                    .and(&forward.prob_e.row(i))
                    .for_each(|mut t_e_row, &f_e| {
                        Zip::from(&mut t_e_row)
                            .and(&backward.prob_e.row(i))
                            .for_each(|t_e, &b_e| {
                                *t_e = f_e + b_e;
                            });
                    });

                #[cfg(feature = "obliv")]
                Zip::from(&mut t)
                    .and(&mut tprobs_e.slice_mut(s![i, .., ..]))
                    .for_each(|t, e| renorm_scale_single(t, e));
            });

        COMB2.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }

    fn first_combine(
        first_c_bprobs: ArrayView2<Real>,
        #[cfg(feature = "obliv")] first_bprobs_e: ArrayView1<TpI16>,
    ) -> Array1<Real> {
        let mut init_tprobs = Zip::from(first_c_bprobs.rows()).map_collect(|r| r.sum());

        #[cfg(feature = "obliv")]
        let mut first_bprobs_e = first_bprobs_e.to_owned();

        #[cfg(feature = "obliv")]
        renorm_equalize_scale_arr1(init_tprobs.view_mut(), first_bprobs_e.view_mut());

        #[cfg(not(feature = "obliv"))]
        {
            init_tprobs *= 1. / init_tprobs.sum();
        }

        let mut init_tprobs_dip = &init_tprobs * &init_tprobs;

        #[cfg(feature = "obliv")]
        {
            init_tprobs_dip /= init_tprobs_dip.sum();
        }

        #[cfg(not(feature = "obliv"))]
        {
            init_tprobs_dip *= 1. / init_tprobs_dip.sum();
        }

        init_tprobs_dip
    }

    fn combine_dip(
        tprobs: ArrayView2<Real>,
        #[cfg(feature = "obliv")] tprobs_e: ArrayView1<TpI16>,
    ) -> Array2<Real> {
        let mut tprobs_dip = Array2::from_shape_fn((P, P), |(i, j)| {
            tprobs[[i, j]] * tprobs[[P - i - 1, P - j - 1]]
        });

        #[cfg(feature = "obliv")]
        {
            let mut tprobs_dip_e = Array1::from_shape_fn(P, |i| tprobs_e[i] + tprobs_e[P - i - 1]);
            let max_e = max_e(tprobs_dip_e.view());
            Zip::from(tprobs_dip.rows_mut())
                .and(&mut tprobs_dip_e)
                .for_each(|t, e| {
                    match_scale_row(max_e, t, e);
                });
            let mut tprobs_dip_sum = tprobs_dip.sum();
            let mut e = TpI16::protect(0);
            renorm_scale_single(&mut tprobs_dip_sum, &mut e);
            tprobs_dip *= Real::protect_i64(1) / tprobs_dip_sum;
            let mut _e = TpI16::protect(0);
            Zip::from(&mut tprobs_dip).for_each(|t| adjust_scale_single(e, t, &mut _e));
        }

        #[cfg(not(feature = "obliv"))]
        {
            tprobs_dip *= 1. / tprobs_dip.sum();
        }

        tprobs_dip
    }
}
