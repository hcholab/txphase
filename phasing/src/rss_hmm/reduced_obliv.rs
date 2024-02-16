use crate::hmm::params::*;

use crate::genotype_graph::{G, P};
use crate::inner::*;
use crate::rss_hmm::filtered_block::FilteredBlockSliceObliv;
use crate::tp_value;
use ndarray::{
    s, Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, ArrayViewMut3, Zip,
};

#[cfg(feature = "obliv")]
use ndarray::ArrayViewMut1;

#[cfg(feature = "obliv")]
use crate::dynamic_fixed::*;

#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::{TpEq, TpI16, TpI8, TpOrd};

#[cfg(feature = "obliv")]
use ndarray::linalg::Dot;
//use tp_fixedpoint::Dot;

#[cfg(not(feature = "obliv"))]
use ndarray::linalg::Dot;

#[cfg(not(feature = "obliv"))]
const RENORM_THRESHOLD: f64 = 1e-20;

#[cfg(not(target_vendor = "fortanix"))]
use std::cell::RefCell;
#[cfg(not(target_vendor = "fortanix"))]
use std::time::{Duration, Instant};
#[cfg(not(target_vendor = "fortanix"))]
thread_local! {
    pub static EMISS: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static TRANS: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static COLL: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static COMB1: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static COMB2: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static EXPAND: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static BLOCK: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static PRE: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static POST: RefCell<Duration> = RefCell::new(Duration::ZERO);
}

#[cfg(feature = "obliv")]
macro_rules! cond_assign_array {
    ($cond: expr, $src: expr, $tar: expr) => {
        Zip::from(&mut $tar)
            .and(&$src)
            .for_each(|t, &s| *t = $cond.select(s, *t));
    };
}

#[derive(Clone)]
pub struct ProbBlock {
    pub c_prob: Array3<Real>,
    pub cnr_prob: Array3<Real>,
    #[cfg(feature = "obliv")]
    pub prob_e: Array2<TpI16>,
    pub alpha_pre: Array2<Real>,
    pub alpha_post: Option<Array1<Real>>,
    pub is_pre: Array1<Bool>,
    pub save_alpha_post_cnr: Option<Array2<Real>>,
}

impl ProbBlock {
    pub fn new(n_sites: usize, n_unique: usize, n_haps: usize) -> Self {
        Self {
            c_prob: Array3::<Real>::zeros((n_sites, P, n_unique)),
            cnr_prob: Array3::<Real>::zeros((n_sites, P, n_unique)),
            #[cfg(feature = "obliv")]
            prob_e: Array2::<TpI16>::from_elem((n_sites, P), TpI16::protect(0)),
            alpha_pre: Array2::<Real>::zeros((n_haps, P)),
            alpha_post: None,
            save_alpha_post_cnr: Some(Array2::<Real>::zeros((P, n_unique))),
            #[cfg(feature = "obliv")]
            is_pre: Array1::from_elem(n_sites, Bool::protect(true)),
            #[cfg(not(feature = "obliv"))]
            is_pre: Array1::from_elem(n_sites, true),
        }
    }

    #[cfg(feature = "obliv")]
    pub fn expand(
        &self,
        index_map: ArrayView1<u16>,
        full_filter: ArrayView1<Bool>,
        inv_weights: ArrayView1<Real>,
    ) -> (Array3<Real>, Array2<TpI16>) {
        let mut expanded = Array3::<Real>::from_elem(
            (
                self.n_sites(),
                self.alpha_pre.dim().0,
                self.alpha_pre.dim().1,
            ),
            Real::ZERO,
        );
        let mut expanded_e =
            Array2::<TpI16>::from_elem((self.n_sites(), self.alpha_pre.dim().1), TpI16::protect(0));

        let cr_prob = &self.c_prob - &self.cnr_prob;
        Zip::from(expanded.outer_iter_mut())
            .and(expanded_e.outer_iter_mut())
            .and(cr_prob.outer_iter())
            .and(self.cnr_prob.outer_iter())
            .and(self.prob_e.outer_iter())
            .and(&self.is_pre)
            .for_each(|mut exp, mut exp_e, cr, cnr, e, &is_pre| {
                HmmReduced::expand_prob(
                    index_map,
                    full_filter,
                    inv_weights,
                    cr,
                    e,
                    cnr,
                    e,
                    self.alpha_pre.view(),
                    self.alpha_post.as_ref().unwrap().view(),
                    is_pre,
                    exp.view_mut(),
                    exp_e.view_mut(),
                );
                renorm_scale(exp.reversed_axes(), exp_e);
            });
        (expanded, expanded_e)
    }

    pub fn n_sites(&self) -> usize {
        self.is_pre.len()
    }
}

pub struct HmmReduced {}

#[cfg(feature = "obliv")]
type FwbwOut = (Array1<Real>, Array1<TpI16>, Array3<Real>, Array3<TpI16>);

#[cfg(not(feature = "obliv"))]
type FwbwOut = (Array1<Real>, Array3<Real>);

impl HmmReduced {
    pub fn fwbw<'a>(
        mut abs_start_site_i: usize,
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

        let mut is_first_segment = tp_value!(true, bool);

        let mut full_trans_prob = Array2::<Real>::zeros((n_full_haps, P));

        #[cfg(feature = "obliv")]
        let mut full_trans_prob_e = Array1::from_elem(P, TpI16::protect(0));

        let mut prev_block_prob: Option<ProbBlock> = None;

        for (block_i, block) in blocks.iter().enumerate() {
            let block_n_sites = block.n_sites();
            let block_n_unique_haps = block.n_unique_haps();

            let mut cur_block_prob =
                ProbBlock::new(block_n_sites + 1, block_n_unique_haps, n_full_haps);

            if block_i == 0 {
                let (mut cur_c_prob, mut next_c_prob) = cur_block_prob
                    .c_prob
                    .multi_slice_mut((s![0, .., ..], s![1, .., ..]));

                #[cfg(feature = "obliv")]
                let (mut cur_c_prob_e, mut next_c_prob_e) = cur_block_prob
                    .prob_e
                    .multi_slice_mut((s![0, ..], s![1, ..]));

                let (mut cur_cnr_prob, mut next_cnr_prob) = cur_block_prob
                    .cnr_prob
                    .multi_slice_mut((s![0, .., ..], s![1, .., ..]));

                Self::init_c_prob(
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

                next_c_prob.assign(&cur_c_prob);
                next_cnr_prob.assign(&cur_cnr_prob);

                #[cfg(feature = "obliv")]
                next_c_prob_e.assign(&cur_c_prob_e);

                let do_collapse = genotype_graph[0].is_segment_marker();

                #[cfg(feature = "obliv")]
                let mut next_cnr_prob_e = next_c_prob_e.to_owned();

                #[cfg(feature = "obliv")]
                Self::collapse(
                    do_collapse,
                    is_first_segment,
                    next_c_prob.view_mut(),
                    next_c_prob_e.view_mut(),
                    next_cnr_prob.view_mut(),
                    next_cnr_prob_e.view_mut(),
                    cur_block_prob
                        .save_alpha_post_cnr
                        .as_mut()
                        .unwrap()
                        .view_mut(),
                );

                #[cfg(feature = "obliv")]
                Zip::from(&next_c_prob_e)
                    .and(next_cnr_prob.rows_mut())
                    .and(&mut next_cnr_prob_e)
                    .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));

                #[cfg(not(feature = "obliv"))]
                if do_collapse {
                    Self::collapse(
                        is_first_segment,
                        next_c_prob.view_mut(),
                        next_cnr_prob.view_mut(),
                        cur_block_prob
                            .save_alpha_post_cnr
                            .as_mut()
                            .unwrap()
                            .view_mut(),
                    );
                }

                is_first_segment = !do_collapse;
            } else {
                let mut prev_block_prob = prev_block_prob.unwrap();
                let prev_block = &blocks[block_i - 1];
                let rprob = rprobs_iter.next().unwrap();

                let do_collapse = genotype_graph[site_i].is_segment_marker();

                Self::block_transition_forward(
                    &mut prev_block_prob,
                    &mut cur_block_prob,
                    &prev_block,
                    &block,
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

                is_first_segment = !do_collapse;
            }

            cur_block_prob.is_pre[0] = tp_value!(true, bool);

            site_i += 1;

            for block_site_i in 2..block_n_sites + 1 {
                let (mut prev_c_prob, mut cur_c_prob) = cur_block_prob
                    .c_prob
                    .multi_slice_mut((s![block_site_i - 1, .., ..], s![block_site_i, .., ..]));

                let (mut prev_cnr_prob, mut cur_cnr_prob) = cur_block_prob
                    .cnr_prob
                    .multi_slice_mut((s![block_site_i - 1, .., ..], s![block_site_i, .., ..]));

                #[cfg(feature = "obliv")]
                let (mut prev_c_prob_e, mut cur_c_prob_e) = cur_block_prob
                    .prob_e
                    .multi_slice_mut((s![block_site_i - 1, ..], s![block_site_i, ..]));

                #[cfg(feature = "obliv")]
                let mut prev_cnr_prob_e = prev_c_prob_e.to_owned();
                #[cfg(feature = "obliv")]
                let mut cur_cnr_prob_e = cur_c_prob_e.to_owned();

                let rprob = rprobs_iter.next().unwrap();

                cur_block_prob.is_pre[block_site_i - 1] = is_first_segment;

                let do_collapse = genotype_graph[site_i].is_segment_marker();

                Self::nonblock_transition_forward(
                    prev_c_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    prev_c_prob_e.view_mut(),
                    prev_cnr_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    prev_cnr_prob_e.view_mut(),
                    block,
                    genotype_graph[site_i],
                    do_collapse,
                    #[cfg(not(feature = "obliv"))]
                    eprob,
                    rprob,
                    block_site_i - 1,
                    is_first_segment,
                    cur_c_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    cur_c_prob_e.view_mut(),
                    cur_cnr_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    cur_cnr_prob_e.view_mut(),
                    cur_block_prob
                        .save_alpha_post_cnr
                        .as_mut()
                        .unwrap()
                        .view_mut(),
                );

                #[cfg(feature = "obliv")]
                {
                    is_first_segment = do_collapse.select(TpBool::protect(false), is_first_segment);
                }

                #[cfg(not(feature = "obliv"))]
                if do_collapse {
                    is_first_segment = false;
                }

                site_i += 1;

                #[cfg(feature = "obliv")]
                {
                    Zip::from(&cur_c_prob_e)
                        .and(cur_cnr_prob.rows_mut())
                        .and(&mut cur_cnr_prob_e)
                        .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));
                }
            }

            Self::compute_alpha_post(
                block.index_map.view(),
                cur_block_prob.alpha_pre.view(),
                &mut cur_block_prob.alpha_post,
                &mut cur_block_prob.save_alpha_post_cnr,
            );

            Self::combine_block(
                abs_start_site_i,
                genotype_graph.slice(s![site_i - block_n_sites..site_i]),
                block,
                &cur_block_prob,
                &bprobs[block_i],
                tprobs.slice_mut(s![site_i - block_n_sites..site_i, .., ..]),
                #[cfg(feature = "obliv")]
                tprobs_e.slice_mut(s![site_i - block_n_sites..site_i, .., ..]),
            );

            abs_start_site_i += block_n_sites;

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

        let mut is_first_segment = tp_value!(true, bool);

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
                    let mut cur_c_prob =
                        cur_block_prob
                            .c_prob
                            .slice_mut(s![block_n_sites - 1, .., ..]);

                    #[cfg(feature = "obliv")]
                    let cur_c_prob_e = cur_block_prob.prob_e.slice_mut(s![block_n_sites - 1, ..]);

                    let mut cur_cnr_prob =
                        cur_block_prob
                            .cnr_prob
                            .slice_mut(s![block_n_sites - 1, .., ..]);

                    Self::init_c_prob(
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

                cur_block_prob.is_pre[block_n_sites - 1] = tp_value!(true, bool);

                if site_i > 0 {
                    site_i -= 1;
                }
            }

            is_first_segment = tp_value!(true, bool);

            for block_site_i in (0..block_n_sites - 1).rev() {
                let (prev_c_prob, cur_c_prob) = cur_block_prob
                    .c_prob
                    .multi_slice_mut((s![block_site_i + 1, .., ..], s![block_site_i, .., ..]));

                let (prev_cnr_prob, mut cur_cnr_prob) = cur_block_prob
                    .cnr_prob
                    .multi_slice_mut((s![block_site_i + 1, .., ..], s![block_site_i, .., ..]));

                #[cfg(feature = "obliv")]
                let (prev_c_prob_e, mut cur_c_prob_e) = cur_block_prob
                    .prob_e
                    .multi_slice_mut((s![block_site_i + 1, ..], s![block_site_i, ..]));

                #[cfg(feature = "obliv")]
                let prev_cnr_prob_e = prev_c_prob_e.to_owned();
                #[cfg(feature = "obliv")]
                let mut cur_cnr_prob_e = cur_c_prob_e.to_owned();

                let rprob = rprobs_iter.next().unwrap();

                let do_collapse = genotype_graph[site_i + 1].is_segment_marker();

                Self::nonblock_transition_backward(
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
                    block_site_i,
                    is_first_segment,
                    cur_c_prob,
                    #[cfg(feature = "obliv")]
                    cur_c_prob_e.view_mut(),
                    cur_cnr_prob.view_mut(),
                    #[cfg(feature = "obliv")]
                    cur_cnr_prob_e.view_mut(),
                    cur_block_prob
                        .save_alpha_post_cnr
                        .as_mut()
                        .unwrap()
                        .view_mut(),
                );

                #[cfg(feature = "obliv")]
                {
                    is_first_segment = do_collapse.select(TpBool::protect(false), is_first_segment);
                }

                #[cfg(not(feature = "obliv"))]
                if do_collapse {
                    is_first_segment = false;
                }

                cur_block_prob.is_pre[block_site_i] = is_first_segment;

                #[cfg(feature = "obliv")]
                Zip::from(&cur_c_prob_e)
                    .and(cur_cnr_prob.rows_mut())
                    .and(&mut cur_cnr_prob_e)
                    .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));

                if site_i > 0 {
                    site_i -= 1;
                }
            }

            if block_i == 0 {
                Self::compute_alpha_post(
                    block.index_map.view(),
                    cur_block_prob.alpha_pre.view(),
                    &mut cur_block_prob.alpha_post,
                    &mut cur_block_prob.save_alpha_post_cnr,
                );
            }

            all_prob_blocks.push(cur_block_prob);
        }

        all_prob_blocks.reverse();

        //{
        //let block = blocks.first().unwrap();

        //let (expanded, expanded_e) = all_prob_blocks.first().unwrap().expand(
        //block.index_map.view(),
        //block.full_filter.view(),
        //block.inv_weights.view(),
        //);

        //let mut filtered_expanded = Array3::<Real>::from_elem(
        //(
        //expanded.dim().0,
        //n_full_states.expose() as usize,
        //expanded.dim().2,
        //),
        //Real::ZERO,
        //);

        //Zip::from(expanded.outer_iter())
        //.and(filtered_expanded.outer_iter_mut())
        //.for_each(|expanded, mut filtered_expanded| {
        //let mut j = 0;
        //Zip::from(expanded.rows())
        //.and(&block.full_filter)
        //.for_each(|r, b| {
        //if b.expose() {
        //filtered_expanded.row_mut(j).assign(&r);
        //j += 1;
        //}
        //});
        //});

        //let i = 3;
        //let mut filtered_expanded = filtered_expanded.slice(s![i, .., ..]);
        //filtered_expanded.swap_axes(0, 1);
        //let filtered_expanded_e = expanded_e.slice(s![i, ..]);
        //let filtered_expanded_exposed =
        //debug_expose_array(filtered_expanded, filtered_expanded_e);

        //let filtered_expanded_exposed =
        //&filtered_expanded_exposed / filtered_expanded_exposed.sum();

        //println!("rss");
        //println!("{:#?}", filtered_expanded_exposed.row(0));
        //println!("rss");
        //println!(
        //"{:#?}",
        //filtered_expanded.row(0).map(|v| v.expose_into_f32())
        //);
        //println!("rss");
        //println!("{:#?}", filtered_expanded_e.map(|v| v.expose()));
        //}

        all_prob_blocks
    }

    fn init_c_prob<'a>(
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

    fn nonblock_transition_forward<'a, 'b>(
        mut prev_c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut prev_c_prob_e: ArrayViewMut1<TpI16>,
        mut prev_cnr_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut prev_cnr_prob_e: ArrayViewMut1<TpI16>,
        block: &FilteredBlockSliceObliv<'a>,
        graph_pos: G,
        do_collapse: Bool,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        rprob: Real,
        block_site_i: usize,
        is_first_segment: Bool,
        mut cur_c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_c_prob_e: ArrayViewMut1<TpI16>,
        mut cur_cnr_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_cnr_prob_e: ArrayViewMut1<TpI16>,
        save_cnr_prob: ArrayViewMut2<Real>,
    ) {
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

        prev_c_prob.assign(&cur_c_prob);
        prev_cnr_prob.assign(&cur_cnr_prob);

        #[cfg(feature = "obliv")]
        {
            prev_c_prob_e.assign(&cur_c_prob_e);
            prev_cnr_prob_e.assign(&cur_cnr_prob_e);
            Zip::from(&prev_c_prob_e)
                .and(prev_cnr_prob.rows_mut())
                .and(&mut prev_cnr_prob_e)
                .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));
        }

        #[cfg(feature = "obliv")]
        Self::collapse(
            do_collapse,
            is_first_segment,
            cur_c_prob.view_mut(),
            cur_c_prob_e.view_mut(),
            cur_cnr_prob.view_mut(),
            cur_cnr_prob_e.view_mut(),
            save_cnr_prob,
        );

        #[cfg(not(feature = "obliv"))]
        if do_collapse {
            Self::collapse(
                is_first_segment,
                cur_c_prob.view_mut(),
                cur_cnr_prob.view_mut(),
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

    fn nonblock_transition_backward<'a, 'b>(
        prev_c_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_c_prob_e: ArrayView1<TpI16>,
        prev_cnr_prob: ArrayView2<Real>,
        #[cfg(feature = "obliv")] prev_cnr_prob_e: ArrayView1<TpI16>,
        block: &FilteredBlockSliceObliv<'a>,
        graph_pos: G,
        do_collapse: Bool,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        rprob: Real,
        block_site_i: usize,
        is_first_segment: Bool,
        mut cur_c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_c_prob_e: ArrayViewMut1<TpI16>,
        mut cur_cnr_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut cur_cnr_prob_e: ArrayViewMut1<TpI16>,
        save_cnr_prob: ArrayViewMut2<Real>,
    ) {
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

        #[cfg(feature = "obliv")]
        Self::collapse(
            do_collapse,
            is_first_segment,
            cur_c_prob.view_mut(),
            cur_c_prob_e.view_mut(),
            cur_cnr_prob.view_mut(),
            cur_cnr_prob_e.view_mut(),
            save_cnr_prob,
        );

        #[cfg(not(feature = "obliv"))]
        if do_collapse {
            Self::collapse(
                is_first_segment,
                cur_c_prob.view_mut(),
                cur_cnr_prob.view_mut(),
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
        prev_block_prob: &mut ProbBlock,
        cur_prob_block: &mut ProbBlock,
        prev_block: &FilteredBlockSliceObliv<'a>,
        cur_block: &FilteredBlockSliceObliv<'a>,
        cur_graph_pos: G,
        do_collapse: Bool,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        rprob: Real,
        mut is_first_segment: Bool,
        mut full_trans_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut full_trans_prob_e: ArrayViewMut1<TpI16>,
    ) {
        let prev_weights = prev_block.weights.view();
        let prev_inv_weights = prev_block.inv_weights.view();
        let prev_index_map = prev_block.index_map.view();
        let prev_full_filter = prev_block.full_filter.view();

        let prev_c_prob = prev_block_prob.c_prob.slice(s![0, .., ..]);
        let prev_cnr_prob = prev_block_prob.cnr_prob.slice(s![0, .., ..]);

        #[cfg(feature = "obliv")]
        let prev_c_prob_e = prev_block_prob.prob_e.row(0);
        #[cfg(feature = "obliv")]
        let prev_cnr_prob_e = prev_block_prob.prob_e.row(0);

        let prev_alpha_pre = prev_block_prob.alpha_pre.view();

        let cur_block_n_sites = cur_block.n_sites();

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
        {
            Self::collapse(
                do_collapse,
                is_first_segment,
                trans_c_prob.view_mut(),
                trans_c_prob_e.view_mut(),
                trans_cnr_prob.view_mut(),
                trans_cnr_prob_e.view_mut(),
                prev_block_prob
                    .save_alpha_post_cnr
                    .as_mut()
                    .unwrap()
                    .view_mut(),
            );

            is_first_segment = do_collapse.select(Bool::protect(false), is_first_segment);
        }

        #[cfg(not(feature = "obliv"))]
        if do_collapse {
            Self::collapse(
                is_first_segment,
                trans_c_prob.view_mut(),
                trans_cnr_prob.view_mut(),
                prev_block_prob
                    .save_alpha_post_cnr
                    .as_mut()
                    .unwrap()
                    .view_mut(),
            );

            is_first_segment = false;
        }

        #[cfg(feature = "obliv")]
        Zip::from(&trans_c_prob_e)
            .and(trans_cnr_prob.rows_mut())
            .and(&mut trans_cnr_prob_e)
            .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));

        let trans_cr_prob = &trans_c_prob - &trans_cnr_prob;

        #[cfg(feature = "obliv")]
        let trans_cr_prob_e = trans_c_prob_e.clone();

        Self::compute_alpha_post(
            prev_index_map,
            prev_alpha_pre.view(),
            &mut prev_block_prob.alpha_post,
            &mut prev_block_prob.save_alpha_post_cnr,
        );

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
            prev_block_prob.alpha_post.as_ref().unwrap().view(),
            is_first_segment,
            full_trans_prob.view_mut(),
            #[cfg(feature = "obliv")]
            full_trans_prob_e.view_mut(),
        );

        #[cfg(not(target_vendor = "fortanix"))]
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

        #[cfg(not(target_vendor = "fortanix"))]
        BLOCK.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        Self::compute_alpha_pre(
            cur_index_map,
            cur_c_prob_.view(),
            full_trans_prob.view_mut(),
            cur_alpha_pre.view_mut(),
        );

        cur_c_prob.assign(&cur_c_prob_.t());

        #[cfg(feature = "obliv")]
        cur_c_prob_e.assign(&trans_c_prob_e);

        #[cfg(feature = "obliv")]
        renorm_scale(cur_c_prob.view_mut(), cur_c_prob_e.view_mut());

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
        prev_block_prob: &mut ProbBlock,
        cur_prob_block: &mut ProbBlock,
        prev_block: &FilteredBlockSliceObliv<'a>,
        cur_block: &FilteredBlockSliceObliv<'a>,
        cur_graph_pos: G,
        do_collapse: Bool,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        rprob: Real,
        is_first_segment: Bool,
        mut full_trans_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut full_trans_prob_e: ArrayViewMut1<TpI16>,
    ) {
        let prev_weights = prev_block.weights.view();
        let prev_inv_weights = prev_block.inv_weights.view();
        let prev_index_map = prev_block.index_map.view();
        let prev_full_filter = prev_block.full_filter.view();

        let prev_block_n_sites = prev_block.n_sites();

        let prev_c_prob = prev_block_prob.c_prob.slice(s![prev_block_n_sites, .., ..]);
        let prev_cnr_prob = prev_block_prob
            .cnr_prob
            .slice(s![prev_block_n_sites, .., ..]);

        #[cfg(feature = "obliv")]
        let prev_c_prob_e = prev_block_prob.prob_e.row(prev_block_n_sites);

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
            prev_c_prob_e.view(),
            trans_c_prob.view_mut(),
            #[cfg(feature = "obliv")]
            trans_c_prob_e.view_mut(),
            trans_cnr_prob.view_mut(),
            #[cfg(feature = "obliv")]
            trans_cnr_prob_e.view_mut(),
        );

        #[cfg(feature = "obliv")]
        Zip::from(&trans_c_prob_e)
            .and(trans_cnr_prob.rows_mut())
            .and(&mut trans_cnr_prob_e)
            .for_each(|&c_e, cnr, cnr_e| match_scale_row(c_e, cnr, cnr_e));

        let trans_cr_prob = &trans_c_prob - &trans_cnr_prob;

        #[cfg(feature = "obliv")]
        let trans_cr_prob_e = trans_c_prob_e.clone();

        Self::compute_alpha_post(
            prev_index_map,
            prev_block_prob.alpha_pre.view(),
            &mut prev_block_prob.alpha_post,
            &mut prev_block_prob.save_alpha_post_cnr,
        );

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
            prev_block_prob.alpha_pre.view(),
            prev_block_prob.alpha_post.as_ref().unwrap().view(),
            is_first_segment,
            full_trans_prob.view_mut(),
            #[cfg(feature = "obliv")]
            full_trans_prob_e.view_mut(),
        );

        #[cfg(not(target_vendor = "fortanix"))]
        let t = Instant::now();

        let (mut cur_c_prob, mut next_c_prob) = cur_prob_block
            .c_prob
            .multi_slice_mut((s![0, .., ..], s![1, .., ..]));

        #[cfg(feature = "obliv")]
        let (mut cur_c_prob_e, mut next_c_prob_e) = cur_prob_block
            .prob_e
            .multi_slice_mut((s![0, ..], s![1, ..]));

        let (mut cur_cnr_prob, mut next_cnr_prob) = cur_prob_block
            .cnr_prob
            .multi_slice_mut((s![0, .., ..], s![1, .., ..]));

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

        cur_cnr_prob.assign(&cur_c_prob);

        #[cfg(not(target_vendor = "fortanix"))]
        BLOCK.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        Self::compute_alpha_pre(
            cur_index_map,
            cur_c_prob_.view(),
            full_trans_prob.view_mut(),
            cur_prob_block.alpha_pre.view_mut(),
        );

        next_c_prob.assign(&cur_c_prob);
        next_cnr_prob.assign(&cur_cnr_prob);

        #[cfg(feature = "obliv")]
        next_c_prob_e.assign(&cur_c_prob_e);

        #[cfg(feature = "obliv")]
        let mut next_cnr_prob_e = next_c_prob_e.to_owned();

        #[cfg(feature = "obliv")]
        Self::collapse(
            do_collapse,
            TpBool::protect(true),
            next_c_prob.view_mut(),
            next_c_prob_e.view_mut(),
            next_cnr_prob.view_mut(),
            next_cnr_prob_e.view_mut(),
            cur_prob_block
                .save_alpha_post_cnr
                .as_mut()
                .unwrap()
                .view_mut(),
        );

        #[cfg(not(feature = "obliv"))]
        if do_collapse {
            Self::collapse(
                true,
                next_c_prob.view_mut(),
                next_cnr_prob.view_mut(),
                cur_prob_block
                    .save_alpha_post_cnr
                    .as_mut()
                    .unwrap()
                    .view_mut(),
            );
        }
        Self::emission(
            cur_block.expand_pos(0).view(),
            cur_graph_pos,
            #[cfg(not(feature = "obliv"))]
            eprob,
            next_c_prob.view_mut(),
            #[cfg(feature = "obliv")]
            next_c_prob_e.view_mut(),
        );
        Self::emission(
            cur_block.expand_pos(0).view(),
            cur_graph_pos,
            #[cfg(not(feature = "obliv"))]
            eprob,
            next_cnr_prob.view_mut(),
            #[cfg(feature = "obliv")]
            next_cnr_prob_e.view_mut(),
        );

        #[cfg(feature = "obliv")]
        Zip::from(&next_c_prob_e)
            .and(next_cnr_prob.rows_mut())
            .and(&mut next_cnr_prob_e)
            .for_each(|&e_to_match, r, e| match_scale_row(e_to_match, r, e));
    }

    fn emission(
        cond_haps: ArrayView1<i8>,
        graph_col: G,
        #[cfg(not(feature = "obliv"))] eprob: Real,
        mut probs: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut probs_e: ArrayViewMut1<TpI16>,
    ) {
        #[cfg(not(target_vendor = "fortanix"))]
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

        #[cfg(not(target_vendor = "fortanix"))]
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
        #[cfg(not(target_vendor = "fortanix"))]
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

        #[cfg(not(target_vendor = "fortanix"))]
        TRANS.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }

    fn collapse(
        #[cfg(feature = "obliv")] do_collapse: Bool,
        do_save_alpha_post_cnr: Bool,
        cur_c_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] cur_c_prob_e: ArrayViewMut1<TpI16>,
        cur_cnr_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] cur_cnr_prob_e: ArrayViewMut1<TpI16>,
        mut save_cnr_prob: ArrayViewMut2<Real>,
    ) {
        #[cfg(not(target_vendor = "fortanix"))]
        let t = Instant::now();

        #[cfg(not(feature = "obliv"))]
        {
            let mut cur_c_prob = cur_c_prob;
            let mut cur_cnr_prob = cur_cnr_prob;
            if do_save_alpha_post_cnr {
                save_cnr_prob.assign(&cur_cnr_prob);
            }
            let mut sum_c = Zip::from(cur_c_prob.columns()).map_collect(|c| c.sum());
            let mut sum_cnr = Zip::from(cur_cnr_prob.columns()).map_collect(|c| c.sum());
            let sum = sum_c.sum();
            if sum < RENORM_THRESHOLD {
                sum_c *= 1. / sum;
                sum_cnr *= 1. / sum;
            }
            Zip::from(cur_c_prob.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_c));
            Zip::from(cur_cnr_prob.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_cnr));
        }

        #[cfg(feature = "obliv")]
        {
            let mut cur_c_prob_ = cur_c_prob;
            let mut cur_c_prob_e_ = cur_c_prob_e;
            let mut cur_cnr_prob_ = cur_cnr_prob;
            let mut cur_cnr_prob_e_ = cur_cnr_prob_e;

            let mut cur_c_prob = cur_c_prob_.to_owned();
            let mut cur_c_prob_e = cur_c_prob_e_.to_owned();
            let mut cur_cnr_prob = cur_cnr_prob_.to_owned();
            let mut cur_cnr_prob_e = cur_cnr_prob_e_.to_owned();

            let (sum_c, sum_c_e) =
                sum_scale_by_column_mut(cur_c_prob.view_mut(), cur_c_prob_e.view_mut());

            let mut sum_cnr_e = max_e(cur_cnr_prob_e.view());
            match_scale(
                sum_cnr_e,
                cur_cnr_prob.view_mut(),
                cur_cnr_prob_e.view_mut(),
            );
            let mut sum_cnr = Zip::from(cur_cnr_prob.columns()).map_collect(|c| c.sum());

            cond_assign_array!(do_save_alpha_post_cnr, cur_cnr_prob, save_cnr_prob);

            renorm_scale_row(sum_cnr.view_mut(), &mut sum_cnr_e);
            Zip::from(cur_c_prob.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_c));
            Zip::from(cur_cnr_prob.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_cnr));
            cur_c_prob_e.fill(sum_c_e);
            cur_cnr_prob_e.fill(sum_cnr_e);
            renorm_e_pair(cur_c_prob_e.view_mut(), cur_cnr_prob_e.view_mut());

            cond_assign_array!(do_collapse, cur_c_prob, cur_c_prob_);
            cond_assign_array!(do_collapse, cur_c_prob_e, cur_c_prob_e_);
            cond_assign_array!(do_collapse, cur_cnr_prob, cur_cnr_prob_);
            cond_assign_array!(do_collapse, cur_cnr_prob_e, cur_cnr_prob_e_);
        }

        #[cfg(not(target_vendor = "fortanix"))]
        COLL.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }

    fn compute_alpha_pre(
        index_map: ArrayView1<u16>,
        c_prob: ArrayView2<Real>,
        full_trans_prob: ArrayViewMut2<Real>,
        mut alpha_pre: ArrayViewMut2<Real>,
    ) {
        #[cfg(not(target_vendor = "fortanix"))]
        let t = Instant::now();

        #[cfg(feature = "obliv")]
        let (div_c_prob, div_c_prob_e) = Self::rev_array_2(c_prob);

        #[cfg(not(feature = "obliv"))]
        let div_c_prob = c_prob.map(|&v| if v == 0. { 0. } else { 1. / v });

        let mut div_c_prob_full = Array2::<Real>::from_elem(full_trans_prob.dim(), Real::ZERO);

        Zip::from(div_c_prob_full.rows_mut())
            .and(&index_map)
            .for_each(|mut r, i| r.assign(&div_c_prob.row(*i as usize)));

        alpha_pre.assign(&(&full_trans_prob * div_c_prob_full));

        #[cfg(feature = "obliv")]
        Zip::from(alpha_pre.rows_mut())
            .and(&index_map)
            .for_each(|mut r, &i| {
                Zip::from(&mut r)
                    .and(&div_c_prob_e.row(i as usize))
                    .for_each(|p, &e| adjust_scale_single(e, p, &mut TpI16::protect(0)));
            });

        #[cfg(not(target_vendor = "fortanix"))]
        PRE.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }

    fn compute_alpha_post(
        index_map: ArrayView1<u16>,
        alpha_pre: ArrayView2<Real>,
        alpha_post: &mut Option<Array1<Real>>,
        save_cnr_prob: &mut Option<Array2<Real>>,
    ) {
        if let (None, Some(save_cnr_prob)) = (alpha_post.as_ref(), save_cnr_prob.take()) {
            #[cfg(not(target_vendor = "fortanix"))]
            let t = Instant::now();
            #[cfg(feature = "obliv")]
            {
                *alpha_post = Some(Array1::<Real>::from_elem(alpha_pre.nrows(), Real::ZERO));
            }

            #[cfg(not(feature = "obliv"))]
            {
                *alpha_post = Some(Array1::<Real>::zeros(alpha_pre.nrows()));
            }
            let mut alpha_post = alpha_post.as_mut().unwrap().view_mut();

            let sum_cnr = Zip::from(save_cnr_prob.columns()).map_collect(|c| c.sum());

            #[cfg(feature = "obliv")]
            let weighted_cur_cnr_prob = {
                let (div_sum_cnr, div_sum_cnr_e) = Self::rev_array_1(sum_cnr.view());

                let mut weighted_cur_cnr_prob = save_cnr_prob.to_owned();
                Zip::from(weighted_cur_cnr_prob.rows_mut()).for_each(|mut r| {
                    r.assign(&(&r * &div_sum_cnr));
                    Zip::from(&mut r).and(&div_sum_cnr_e).for_each(|p, &e| {
                        adjust_scale_single(e, p, &mut TpI16::protect(0));
                    });
                });

                Array::from_shape_vec(
                    weighted_cur_cnr_prob.t().raw_dim(),
                    weighted_cur_cnr_prob.t().iter().cloned().collect(),
                )
                .unwrap()
            };

            Zip::from(&mut alpha_post)
                .and(alpha_pre.rows())
                .and(&index_map)
                .for_each(|a_post, a_pre, &i| {
                    let i = i as usize;

                    #[cfg(feature = "obliv")]
                    {
                        *a_post = sum_cnr[[i]]
                            .tp_eq(&Real::ZERO)
                            .select(Real::ZERO, Dot::dot(&weighted_cur_cnr_prob.row(i), &a_pre));
                    }

                    #[cfg(not(feature = "obliv"))]
                    {
                        *a_post = if sum_cnr[[i]] == 0. {
                            0.
                        } else {
                            save_cnr_prob.column(i).dot(&a_pre) / sum_cnr[i]
                        };
                    }
                });
            #[cfg(not(target_vendor = "fortanix"))]
            POST.with(|v| {
                let mut v = v.borrow_mut();
                *v += t.elapsed();
            });
        }
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
        is_pre: Bool,
        mut expanded_prob: ArrayViewMut2<Real>,
        #[cfg(feature = "obliv")] mut expanded_prob_e: ArrayViewMut1<TpI16>,
    ) {
        #[cfg(not(target_vendor = "fortanix"))]
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

        #[cfg(feature = "obliv")]
        {
            let alpha = Self::select_alpha(is_pre, alpha_pre.view(), alpha_post.view());

            let mut cr_prob_full = Array2::<Real>::from_elem(alpha.dim(), Real::ZERO);
            Zip::from(cr_prob_full.rows_mut())
                .and(&index_map)
                .for_each(|mut r, &i| r.assign(&cr_prob.row(i as usize)));

            let mut cnr_prob_full = Array2::<Real>::from_elem(alpha.dim(), Real::ZERO);
            Zip::from(cnr_prob_full.rows_mut())
                .and(&index_map)
                .for_each(|mut r, &i| r.assign(&cnr_prob.row(i as usize)));

            expanded_prob.assign(&(cr_prob_full + cnr_prob_full * alpha));

            Zip::from(expanded_prob.rows_mut())
                .and(&full_filter)
                .for_each(|mut r, &b| {
                    Zip::from(&mut r).for_each(|p| {
                        *p = b.select(*p, Real::ZERO);
                    })
                });
        }

        #[cfg(not(feature = "obliv"))]
        if is_pre {
            Zip::from(expanded_prob.rows_mut())
                .and(alpha_pre.rows())
                .and(&index_map)
                .and(&full_filter)
                .for_each(|mut p, a, &i, &b| {
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

        #[cfg(not(target_vendor = "fortanix"))]
        EXPAND.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }
    fn combine_block<'a>(
        start_site_i: usize,
        genotype_graph: ArrayView1<G>,
        block: &FilteredBlockSliceObliv<'a>,
        forward: &ProbBlock,
        backward: &ProbBlock,
        mut tprobs: ArrayViewMut3<Real>,
        #[cfg(feature = "obliv")] mut tprobs_e: ArrayViewMut3<TpI16>,
    ) {
        #[cfg(not(target_vendor = "fortanix"))]
        let t = Instant::now();
        let inv_weights = block.inv_weights.view();

        let n_unique_haps = block.n_unique_haps();

        let mut alpha_11 = Array3::<Real>::zeros((P, P, n_unique_haps));
        let mut alpha_10 = Array2::<Real>::zeros((P, n_unique_haps));
        let mut alpha_01 = Array2::<Real>::zeros((P, n_unique_haps));
        let mut alpha_00 = Array1::<Real>::zeros(n_unique_haps);

        let forward_alpha_post = forward.alpha_post.as_ref().unwrap().view();
        let backward_alpha_post = backward.alpha_post.as_ref().unwrap().view();

        let mut rev_index_map = vec![Vec::new(); n_unique_haps];
        for (i, &j) in block.index_map.iter().enumerate() {
            rev_index_map[j as usize].push(i);
        }

        for (i, indices) in rev_index_map.into_iter().enumerate() {
            let mut f_pre_arr = Array2::<Real>::zeros((P, indices.len()));
            let mut f_post_arr = Array1::<Real>::zeros(indices.len());
            let mut b_pre_arr = Array2::<Real>::zeros((P, indices.len()));
            let mut b_post_arr = Array1::<Real>::zeros(indices.len());

            for (j, k) in indices.into_iter().enumerate() {
                f_pre_arr.column_mut(j).assign(&forward.alpha_pre.row(k));
                f_post_arr[j] = forward_alpha_post[k];
                b_pre_arr.column_mut(j).assign(&backward.alpha_pre.row(k));
                b_post_arr[j] = backward_alpha_post[k];
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

        #[cfg(not(target_vendor = "fortanix"))]
        COMB1.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        #[cfg(not(target_vendor = "fortanix"))]
        let t = Instant::now();

        #[cfg(not(feature = "obliv"))]
        {
            Zip::indexed(tprobs.outer_iter_mut())
                .and(
                    forward
                        .c_prob
                        .slice(s![..block.n_sites(), .., ..])
                        .outer_iter(),
                )
                .and(
                    forward
                        .cnr_prob
                        .slice(s![..block.n_sites(), .., ..])
                        .outer_iter(),
                )
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
                });
        }

        #[cfg(feature = "obliv")]
        {
            let mut fc_save: Option<Array2<Real>> = None;
            let mut fcnr_save: Option<Array2<Real>> = None;
            let mut bc_save: Option<Array2<Real>> = None;
            let mut bcnr_save: Option<Array2<Real>> = None;
            let mut f_e_save: Option<Array1<TpI16>> = None;
            let mut b_e_save: Option<Array1<TpI16>> = None;
            let mut f_is_pre = Bool::protect(false);
            let mut b_is_pre = Bool::protect(false);
            let mut abs_site_i = start_site_i;

            Zip::indexed(
                forward
                    .c_prob
                    .slice(s![..block.n_sites(), .., ..])
                    .outer_iter(),
            )
            .and(
                forward
                    .cnr_prob
                    .slice(s![..block.n_sites(), .., ..])
                    .outer_iter(),
            )
            .and(backward.c_prob.outer_iter())
            .and(backward.cnr_prob.outer_iter())
            .for_each(|i, fc, fcnr, bc, bcnr| {
                let cond = genotype_graph[i].is_segment_marker();
                if let Some(fc_save) = fc_save.as_mut() {
                    cond_assign_array!(cond, fc, *fc_save);
                } else {
                    fc_save = Some(fc.to_owned());
                }
                if let Some(fcnr_save) = fcnr_save.as_mut() {
                    cond_assign_array!(cond, fcnr, *fcnr_save);
                } else {
                    fcnr_save = Some(fcnr.to_owned());
                }
                if let Some(bc_save) = bc_save.as_mut() {
                    cond_assign_array!(cond, bc, *bc_save);
                } else {
                    bc_save = Some(bc.to_owned());
                }
                if let Some(bcnr_save) = bcnr_save.as_mut() {
                    cond_assign_array!(cond, bcnr, *bcnr_save);
                } else {
                    bcnr_save = Some(bcnr.to_owned());
                }

                if let Some(f_e_save) = f_e_save.as_mut() {
                    cond_assign_array!(cond, forward.prob_e.row(i), *f_e_save);
                } else {
                    f_e_save = Some(forward.prob_e.row(i).to_owned());
                }

                if let Some(b_e_save) = b_e_save.as_mut() {
                    cond_assign_array!(cond, backward.prob_e.row(i), *b_e_save);
                } else {
                    b_e_save = Some(backward.prob_e.row(i).to_owned());
                }

                f_is_pre = cond.select(forward.is_pre[i], f_is_pre);
                b_is_pre = cond.select(backward.is_pre[i], b_is_pre);

                if abs_site_i % 3 == 0 || i == block.n_sites() - 1 {
                    let fc = fc_save.take().unwrap();
                    let fcnr = fcnr_save.take().unwrap();
                    let bc = bc_save.take().unwrap();
                    let bcnr = bcnr_save.take().unwrap();
                    let f_e = f_e_save.take().unwrap();
                    let b_e = b_e_save.take().unwrap();

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
                    let alpha = {
                        let alpha_backward_0 =
                            Self::select_alpha_2(b_is_pre, alpha_01.view(), alpha_00.view());

                        let alpha_backward_1 =
                            Self::select_alpha_3(b_is_pre, alpha_11.view(), alpha_10.view());

                        Self::select_alpha_3(
                            f_is_pre,
                            alpha_backward_1.view(),
                            alpha_backward_0.view(),
                        )
                    };

                    let mut t = tprobs.slice(s![i, .., ..]).to_owned();
                    let mut t_e = tprobs_e.slice(s![i, .., ..]).to_owned();

                    Zip::from(alpha.outer_iter())
                        .and(fcnr.outer_iter())
                        .and(t.rows_mut())
                        .for_each(|a1, f, mut t_r| {
                            Zip::from(a1.outer_iter())
                                .and(bcnr.outer_iter())
                                .and(&mut t_r)
                                .for_each(|a2, b, t_i| {
                                    *t_i = Dot::dot(&(&a2 * &f), &b.t());
                                });
                        });
                    t += &c;

                    Zip::from(t_e.rows_mut())
                        .and(&f_e)
                        .for_each(|mut t_e_row, &f_e_i| {
                            Zip::from(&mut t_e_row).and(&b_e).for_each(|t_e, &b_e_i| {
                                *t_e = f_e_i + b_e_i;
                            });
                        });

                    Zip::from(&mut t)
                        .and(&mut t_e)
                        .for_each(|t, e| renorm_scale_single(t, e));

                    let n_write_back = (i + 1).min(if abs_site_i % 3 == 0 {
                        3
                    } else {
                        abs_site_i % 3
                    });
                    for j in 0..n_write_back {
                        tprobs.slice_mut(s![i - j, .., ..]).assign(&t);
                        tprobs_e.slice_mut(s![i - j, .., ..]).assign(&t_e);
                    }

                    f_is_pre = Bool::protect(false);
                    b_is_pre = Bool::protect(false);
                }
                abs_site_i += 1;
            });
        }

        #[cfg(not(target_vendor = "fortanix"))]
        COMB2.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });
    }

    //fn first_combine(
    //first_c_bprobs: ArrayView2<Real>,
    //#[cfg(feature = "obliv")] first_bprobs_e: ArrayView1<TpI16>,
    //) -> Array1<Real> {
    //let mut init_tprobs = Zip::from(first_c_bprobs.rows()).map_collect(|r| r.sum());

    //#[cfg(feature = "obliv")]
    //let mut first_bprobs_e = first_bprobs_e.to_owned();

    //#[cfg(feature = "obliv")]
    //renorm_equalize_scale_arr1(init_tprobs.view_mut(), first_bprobs_e.view_mut());

    //#[cfg(not(feature = "obliv"))]
    //{
    //init_tprobs *= 1. / init_tprobs.sum();
    //}

    //let mut init_tprobs_dip = &init_tprobs * &init_tprobs;

    //#[cfg(feature = "obliv")]
    //{
    //init_tprobs_dip /= init_tprobs_dip.sum();
    //}

    //#[cfg(not(feature = "obliv"))]
    //{
    //init_tprobs_dip *= 1. / init_tprobs_dip.sum();
    //}

    //init_tprobs_dip
    //}

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

    #[cfg(feature = "obliv")]
    fn select_alpha(
        is_pre: Bool,
        alpha_pre: ArrayView2<Real>,
        alpha_post: ArrayView1<Real>,
    ) -> Array2<Real> {
        let mut alpha = Array2::from_elem(alpha_pre.dim(), Real::ZERO);

        Zip::from(alpha.rows_mut())
            .and(alpha_pre.rows())
            .and(&alpha_post)
            .for_each(|mut a_row, pre_row, &post| {
                Zip::from(&mut a_row).and(&pre_row).for_each(|a, &pre| {
                    *a = is_pre.select(pre, post);
                });
            });

        alpha
    }

    #[cfg(feature = "obliv")]
    fn select_alpha_2(
        is_pre: Bool,
        alpha_pre: ArrayView2<Real>,
        alpha_post: ArrayView1<Real>,
    ) -> Array2<Real> {
        let mut alpha = Array2::from_elem(alpha_pre.dim(), Real::ZERO);

        Zip::from(alpha.columns_mut())
            .and(alpha_pre.columns())
            .and(&alpha_post)
            .for_each(|mut a_row, pre_row, &post| {
                Zip::from(&mut a_row).and(&pre_row).for_each(|a, &pre| {
                    *a = is_pre.select(pre, post);
                });
            });

        alpha
    }

    #[cfg(feature = "obliv")]
    fn select_alpha_3(
        is_pre: Bool,
        mut alpha_pre: ndarray::ArrayView3<Real>,
        alpha_post: ArrayView2<Real>,
    ) -> Array3<Real> {
        let mut alpha = ndarray::Array3::from_elem(alpha_pre.dim(), Real::ZERO);
        alpha.swap_axes(0, 1);
        alpha_pre.swap_axes(0, 1);

        Zip::from(alpha.outer_iter_mut())
            .and(alpha_pre.outer_iter())
            .for_each(|mut alpha, alpha_pre| {
                Zip::from(&mut alpha)
                    .and(&alpha_pre)
                    .and(&alpha_post)
                    .for_each(|a, &pre, &post| *a = is_pre.select(pre, post))
            });

        alpha
    }

    #[cfg(feature = "obliv")]
    fn rev_array_1(array: ArrayView1<Real>) -> (Array1<Real>, Array1<TpI16>) {
        let mut cur = Real::protect_i64(1);
        let mut cur_e = TpI16::protect(0);
        let mut rev = Array1::from_elem(array.dim(), Real::ZERO);
        let mut rev_e = Array1::from_elem(array.dim(), TpI16::protect(0));

        let mut array = array.to_owned();
        let mut array_e = Array1::from_elem(array.dim(), TpI16::protect(0));
        Zip::from(&mut array)
            .and(&mut array_e)
            .for_each(|p, e| renorm_scale_single(p, e));

        for ((&a, &a_e), (r, r_e)) in array
            .iter()
            .zip(array_e.iter())
            .zip(rev.iter_mut().zip(rev_e.iter_mut()))
        {
            let cond = a.tp_not_eq(&0);
            *r = cond.select(cur, Real::ZERO);
            *r_e = cond.select(cur_e, TpI16::protect(0));
            cur = cond.select(cur * a, cur);
            cur_e = cond.select(cur_e + a_e, cur_e);
            renorm_scale_single(&mut cur, &mut cur_e);
        }
        let prod = cur;
        let prod_e = cur_e;
        cur = Real::protect_i64(1);
        cur_e = TpI16::protect(0);

        for ((&a, &a_e), (r, r_e)) in array
            .as_slice()
            .unwrap()
            .iter()
            .zip(array_e.as_slice().unwrap().iter())
            .zip(
                rev.as_slice_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(rev_e.as_slice_mut().unwrap().iter_mut()),
            )
            .rev()
        {
            let cond = a.tp_not_eq(&0);
            *r = cond.select(*r * cur, Real::ZERO);
            *r_e = cond.select(*r_e + cur_e, TpI16::protect(0));
            cur = cond.select(cur * a, cur);
            cur_e = cond.select(cur_e + a_e, cur_e);
            renorm_scale_single(&mut cur, &mut cur_e);
        }
        let rev_prod = Real::protect_i64(1) / prod;
        let mut out = rev * rev_prod;
        let mut out_e = rev_e.mapv_into(|v| v - prod_e);
        Zip::from(&mut out).and(&mut out_e).for_each(|o, o_e| {
            renorm_scale_single(o, o_e);
            *o_e = o.tp_eq(&0).select(TpI16::protect(0), *o_e);
        });
        (out, -out_e)
    }

    #[cfg(feature = "obliv")]
    fn rev_array_2(array: ArrayView2<Real>) -> (Array2<Real>, Array2<TpI16>) {
        let mut cur = Real::protect_i64(1);
        let mut cur_e = TpI16::protect(0);
        let mut rev = Array2::from_elem(array.dim(), Real::ZERO);
        let mut rev_e = Array2::from_elem(array.dim(), TpI16::protect(0));

        let mut array = array.to_owned();
        let mut array_e = Array2::from_elem(array.dim(), TpI16::protect(0));
        Zip::from(&mut array)
            .and(&mut array_e)
            .for_each(|p, e| renorm_scale_single(p, e));

        for ((&a, &a_e), (r, r_e)) in array
            .iter()
            .zip(array_e.iter())
            .zip(rev.iter_mut().zip(rev_e.iter_mut()))
        {
            let cond = a.tp_not_eq(&0);
            *r = cond.select(cur, Real::ZERO);
            *r_e = cond.select(cur_e, TpI16::protect(0));
            cur = cond.select(cur * a, cur);
            cur_e = cond.select(cur_e + a_e, cur_e);
            renorm_scale_single(&mut cur, &mut cur_e);
        }
        let prod = cur;
        let prod_e = cur_e;
        cur = Real::protect_i64(1);
        cur_e = TpI16::protect(0);

        for ((&a, &a_e), (r, r_e)) in array
            .as_slice()
            .unwrap()
            .iter()
            .zip(array_e.as_slice().unwrap().iter())
            .zip(
                rev.as_slice_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(rev_e.as_slice_mut().unwrap().iter_mut()),
            )
            .rev()
        {
            let cond = a.tp_not_eq(&0);
            *r = cond.select(*r * cur, Real::ZERO);
            *r_e = cond.select(*r_e + cur_e, TpI16::protect(0));
            cur = cond.select(cur * a, cur);
            cur_e = cond.select(cur_e + a_e, cur_e);
            renorm_scale_single(&mut cur, &mut cur_e);
        }
        let rev_prod = Real::protect_i64(1) / prod;
        let mut out = rev * rev_prod;
        let mut out_e = rev_e.mapv_into(|v| v - prod_e);
        Zip::from(&mut out).and(&mut out_e).for_each(|o, o_e| {
            renorm_scale_single(o, o_e);
            *o_e = o.tp_eq(&0).select(TpI16::protect(0), *o_e);
        });
        (out, -out_e)
    }
}
