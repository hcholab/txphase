#[cfg(feature = "obliv")]
use timing_shield::TpOrd;

#[cfg(feature = "obliv")]
use obliv_utils::top_s::{merge_top_s, select_top_s};

#[cfg(not(feature = "obliv"))]
use crate::top_s::{merge_top_s, select_top_s};

use super::{NNRank, RankList};
use crate::pbwt_trie::Node;
use crate::U32;
use std::rc::Rc;

#[cfg(not(target_vendor = "fortanix"))]
mod timing {
    pub use std::cell::RefCell;
    pub use std::time::{Duration, Instant};
    thread_local! {
        pub static CREATE: RefCell<Duration> = RefCell::new(Duration::ZERO);
        pub static SORT: RefCell<Duration> = RefCell::new(Duration::ZERO);
    }
}
#[cfg(not(target_vendor = "fortanix"))]
pub use timing::*;

pub fn init_last_rank_level(
    n_neighbors: usize,
    prev_input_pos: U32,
    init_input_div: &[U32],
    prev_ppa: &[Vec<usize>],
    cur_ppa: &[Vec<usize>],
) -> Vec<Rc<RankList<NNRank>>> {
    #[cfg(not(target_vendor = "fortanix"))]
    let t = Instant::now();

    let len = cur_ppa.iter().map(|v| v.len()).sum::<usize>();
    let mut rev_cur_ppa = vec![0u32; len];
    for (i, g) in cur_ppa.iter().enumerate() {
        for &j in g {
            rev_cur_ppa[j] = i as u32;
        }
    }

    let mut ranks = vec![Vec::new(); cur_ppa.len()];

    prev_ppa
        .into_iter()
        .flatten()
        .enumerate()
        .for_each(|(i, &hap_id)| {
            #[cfg(feature = "obliv")]
            let (dist, is_below) = {
                let i = i as u32;
                let cond = prev_input_pos.tp_gt(&i);
                let dist = cond.select(prev_input_pos - i, i - prev_input_pos + 1);
                let is_below = !cond;
                (dist, is_below)
            };

            #[cfg(not(feature = "obliv"))]
            let (dist, is_below) = if i < prev_input_pos {
                (prev_input_pos - i, false)
            } else {
                (i - prev_input_pos + 1, true)
            };

            ranks[rev_cur_ppa[hap_id] as usize].push(NNRank::new(
                init_input_div[hap_id],
                dist,
                is_below,
                hap_id,
            ));
        });

    #[cfg(not(target_vendor = "fortanix"))]
    CREATE.with(|v| {
        let mut v = v.borrow_mut();
        *v += t.elapsed();
    });

    #[cfg(not(target_vendor = "fortanix"))]
    let t = Instant::now();

    let out = ranks
        .into_iter()
        .map(|r| Rc::new(select_top_s(n_neighbors, r)))
        .collect();

    #[cfg(not(target_vendor = "fortanix"))]
    SORT.with(|v| {
        let mut v = v.borrow_mut();
        *v += t.elapsed();
    });

    out
}

pub fn build_rank_level(
    n_neighbors: usize,
    trie_level: &[Node],
    prev_rank_level: &[Rc<RankList<NNRank>>],
) -> Vec<Rc<RankList<NNRank>>> {
    trie_level
        .iter()
        .map(|&t| match t {
            (Some(a), Some(b)) => Rc::new(merge_top_s(
                n_neighbors,
                &prev_rank_level[a as usize],
                &prev_rank_level[b as usize],
            )),
            (Some(a), None) => prev_rank_level[a as usize].clone(),
            (None, Some(b)) => prev_rank_level[b as usize].clone(),
            (None, None) => panic!("This should never happen!"),
        })
        .collect::<Vec<_>>()
}
