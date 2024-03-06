use super::{InitRank, RankList};
use crate::pbwt_trie::{Node, PbwtTrieInput};

#[cfg(feature = "obliv")]
use obliv_utils::top_s::{merge_top_s, select_top_s_stable};

#[cfg(not(feature = "obliv"))]
use crate::top_s::{merge_top_s, select_top_s};

const INIT_NEIGHBORS: usize = 2;

pub fn build_init_tree(
    trie: &[Vec<Node>],
    input: &PbwtTrieInput,
    ppa: &[Vec<u32>],
    next_first_full_haps: Option<&[bool]>, // (index map, haps)
) -> Vec<Vec<RankList<InitRank>>> {
    let mut init_tree = Vec::new();
    let mut init_level = build_init_tree_leaves(input, ppa, next_first_full_haps);

    for trie_level in trie.iter().rev() {
        init_tree.push(init_level);
        init_level = build_init_tree_level(trie_level, init_tree.last_mut().unwrap());
    }
    init_tree.reverse();
    init_tree
}

fn build_init_tree_leaves(
    prev_input: &PbwtTrieInput,
    ppa: &[Vec<u32>],
    next_first_full_haps: Option<&[bool]>,
) -> Vec<RankList<InitRank>> {
    ppa.iter()
        .map(|group| {
            let ranks = group
                .iter()
                .map(|&hap_id| {
                    InitRank::new(
                        prev_input.full_div[hap_id as usize],
                        get_hap(hap_id, next_first_full_haps),
                    )
                })
                .collect();
            select_top_s_stable(INIT_NEIGHBORS, ranks)
        })
        .collect()
}

fn get_hap(hap_id: u32, haps: Option<&[bool]>) -> bool {
    haps.map(|v| v[hap_id as usize]).unwrap_or(false)
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
