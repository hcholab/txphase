use super::{InitRank, RankList};
use crate::pbwt_trie::{Node, PbwtTrieInput};

#[cfg(feature = "obliv")]
use obliv_utils::top_s::{merge_top_s, select_top_s};

#[cfg(not(feature = "obliv"))]
use crate::top_s::{merge_top_s, select_top_s};

#[cfg(feature = "obliv")]
use timing_shield::TpOrd;

const INIT_NEIGHBORS: usize = 2;

pub fn build_init_tree(
    trie: &[Vec<Node>],
    input: &PbwtTrieInput,
    prev_ppa: Option<&[Vec<usize>]>,
    cur_ppa: &[Vec<usize>],
    next_first_full_haps: Option<&[bool]>, // (index map, haps)
) -> Vec<Vec<RankList<InitRank>>> {
    let mut init_tree = Vec::new();
    let mut init_level = build_init_tree_leaves(input, prev_ppa, cur_ppa, next_first_full_haps);

    for trie_level in trie.iter().rev() {
        init_tree.push(init_level);
        init_level = build_init_tree_level(trie_level, init_tree.last_mut().unwrap());
    }
    init_tree.reverse();
    init_tree
}

fn build_init_tree_leaves(
    prev_input: &PbwtTrieInput,
    prev_ppa: Option<&[Vec<usize>]>,
    cur_ppa: &[Vec<usize>],
    next_first_full_haps: Option<&[bool]>,
) -> Vec<RankList<InitRank>> {
    // if first block, use zero ppa
    let zero_ppa;
    let prev_ppa = prev_ppa.unwrap_or({
        let n_haps = prev_input.full_div.len();
        zero_ppa = vec![(0..n_haps).collect::<Vec<_>>()];
        &zero_ppa
    });

    let ranks = prev_ppa
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, &hap_id)| {
            #[cfg(feature = "obliv")]
            let (dist, is_below) = {
                let i = i as u64;
                let cond = prev_input.full_pos.tp_gt(&i);
                let dist = cond.select(prev_input.full_pos - i, i - prev_input.full_pos + 1);
                let is_below = !cond;
                (dist, is_below)
            };

            #[cfg(not(feature = "obliv"))]
            let (dist, is_below) = if i < prev_input.full_pos {
                (prev_input.full_pos - i, false)
            } else {
                (i - prev_input.full_pos + 1, true)
            };

            (
                hap_id,
                InitRank::new(
                    prev_input.full_div[hap_id],
                    dist,
                    is_below,
                    get_hap(hap_id, next_first_full_haps),
                ),
            )
        })
        .collect::<Vec<_>>();

    let mut sorted_ranks: Vec<Option<InitRank>> = vec![None; ranks.len()];

    for (id, rank) in ranks {
        sorted_ranks[id] = Some(rank);
    }

    cur_ppa
        .into_iter()
        .map(|group| {
            let rank = group
                .into_iter()
                .map(|&id| sorted_ranks[id].take().unwrap())
                .collect::<Vec<_>>();
            select_top_s(INIT_NEIGHBORS, rank)
        })
        .collect()
}

fn get_hap(hap_id: usize, haps: Option<&[bool]>) -> bool {
    haps.map(|v| v[hap_id]).unwrap_or(false)
}

fn build_init_tree_level(
    trie_level: &[Node],
    next_level: &[RankList<InitRank>],
) -> Vec<RankList<InitRank>> {
    trie_level
        .iter()
        .map(|&t| match t {
            (Some(a), Some(b)) => {
                let mut tmp_a = next_level[a as usize].clone();
                let mut tmp_b = next_level[b as usize].clone();
                for v in tmp_a.iter_mut() {
                    v.set_hap(false);
                }
                for v in tmp_b.iter_mut() {
                    v.set_hap(true);
                }
                merge_top_s(INIT_NEIGHBORS, &tmp_a, &tmp_b)
            }
            (Some(a), None) => {
                let mut tmp_a = next_level[a as usize].clone();
                for v in tmp_a.iter_mut() {
                    v.set_hap(false);
                }
                tmp_a
            }
            (None, Some(b)) => {
                let mut tmp_b = next_level[b as usize].clone();
                for v in tmp_b.iter_mut() {
                    v.set_hap(true);
                }
                tmp_b
            }
            (None, None) => panic!("This should never happen!"),
        })
        .collect::<Vec<_>>()
}
