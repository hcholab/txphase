#[cfg(feature = "obliv")]
use timing_shield::TpOrd;

#[cfg(feature = "obliv")]
use obliv_utils::top_s::{merge_top_s, select_top_s};

#[cfg(not(feature = "obliv"))]
use crate::top_s::{merge_top_s, select_top_s};

use super::{NNRank, RankList};
use crate::pbwt_trie::Node;
use crate::Usize;
use std::rc::Rc;

pub fn init_last_rank_level(
    n_neighbors: usize,
    prev_input_pos: Usize,
    init_input_div: &[Usize],
    prev_ppa: &[Vec<usize>],
    cur_ppa: &[Vec<usize>],
) -> Vec<Rc<RankList<NNRank>>> {
    let ranks = prev_ppa
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, &hap_id)| {
            #[cfg(feature = "obliv")]
            let (dist, is_below) = {
                let i = i as u64;
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

            (
                hap_id,
                NNRank::new(init_input_div[hap_id], dist, is_below, hap_id),
            )
        })
        .collect::<Vec<_>>();
    let mut sorted_ranks = vec![None; ranks.len()];
    for (id, rank) in ranks {
        sorted_ranks[id] = Some(rank);
    }

    let ranks = cur_ppa
        .into_iter()
        .map(|group| {
            let rank = group
                .into_iter()
                .map(|&id| sorted_ranks[id].take().unwrap())
                .collect::<Vec<_>>();
            Rc::new(select_top_s(n_neighbors, rank))
        })
        .collect();
    ranks
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
