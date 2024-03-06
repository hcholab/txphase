#[cfg(feature = "obliv")]
use obliv_utils::top_s::{merge_top_s, select_top_s};

#[cfg(not(feature = "obliv"))]
use crate::top_s::{merge_top_s, select_top_s};

use super::{NNRank, RankList};
use crate::pbwt_trie::Node;
use crate::U32;
use std::rc::Rc;

pub fn init_last_rank_level(
    n_neighbors: usize,
    init_input_div: &[U32],
    ppa: &[Vec<u32>],
) -> Vec<Rc<RankList<NNRank>>> {
    ppa.iter()
        .map(|group| {
            let ranks = group
                .into_iter()
                .map(|&hap_id| NNRank::new(init_input_div[hap_id as usize], hap_id))
                .collect::<Vec<_>>();
            Rc::new(select_top_s(n_neighbors, ranks))
        })
        .collect()
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
