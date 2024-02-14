use crate::nn::nn_tree::{build_rank_level, init_last_rank_level};

#[cfg(feature = "obliv")]
use crate::U16;

use super::{NNRank, RankList};

use crate::pbwt_trie::{nearest_group, PbwtTrie, PbwtTrieInput};
use crate::{Bool, Usize};

#[cfg(feature = "obliv")]
use timing_shield::TpEq;

//use std::sync::{Arc, Mutex};
//use std::time::{Duration, Instant};

//lazy_static::lazy_static! {
//pub static ref INIT_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
//pub static ref INSERT_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
//pub static ref NEIGH_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
//pub static ref UPDATE_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
//}

pub fn find_top_neighbors(
    input_hap: &[Bool],
    n_neighbors: usize,
    pbwt_tries: &[PbwtTrie],
    n_haps: usize,
    find_neighbors_filter: &[bool],
) -> Vec<Option<Vec<Usize>>> {
    assert_eq!(input_hap.len(), find_neighbors_filter.len());
    let mut pbwt_input = PbwtTrieInput::new(n_haps);
    let mut prev_ppa = &vec![(0..n_haps).collect::<Vec<_>>()];
    let mut neighbors = Vec::new();
    for pbwt in pbwt_tries {
        let input_slice = &input_hap[pbwt.start_site..pbwt.start_site + pbwt.n_sites()];
        let new_neighbors = find_top_neighbors_trie(
            input_slice,
            n_neighbors,
            pbwt,
            prev_ppa,
            &mut pbwt_input,
            &find_neighbors_filter[pbwt.start_site..pbwt.start_site + pbwt.n_sites()],
        );
        neighbors.extend(new_neighbors.into_iter().rev());
        prev_ppa = &pbwt.ppa;
    }
    neighbors
}

fn find_top_neighbors_trie(
    input_hap: &[Bool],
    n_neighbors: usize,
    pbwt_trie: &PbwtTrie,
    prev_ppa: &[Vec<usize>],
    pbwt_input: &mut PbwtTrieInput,
    find_neighbors_filter: &[bool],
) -> Vec<Option<Vec<Usize>>> {
    #[cfg(feature = "obliv")]
    let (mut last_group_id, mut last_div_above, mut last_div_below) =
        (U16::protect(0), U16::protect(0), U16::protect(0));

    #[cfg(not(feature = "obliv"))]
    let (mut last_group_id, mut last_div_above, mut last_div_below) = (0, 0, 0);

    //let t = Instant::now();
    let mut nearest_neighbors = Vec::new();
    for (i, ((id, d_a, d_b), &b)) in pbwt_trie
        .insert(input_hap.iter().cloned())
        .zip(find_neighbors_filter.into_iter())
        .enumerate()
    {
        nearest_neighbors.push(if b {
            Some(nearest_group(id, d_a, d_b))
        } else {
            None
        });
        if i == input_hap.len() - 1 {
            last_group_id = id;
            last_div_above = d_a;
            last_div_below = d_b;
        }
    }
    let mut nearest_neighbors_iter = nearest_neighbors.into_iter().rev();
    //let mut _t = INSERT_T.lock().unwrap();
    //*_t += Instant::now() - t;

    //let t = Instant::now();
    let mut rank_level = init_last_rank_level(
        n_neighbors,
        pbwt_input.full_pos,
        &pbwt_input.full_div,
        &prev_ppa,
        &pbwt_trie.ppa,
    );
    //let mut _t = INIT_T.lock().unwrap();
    //*_t += Instant::now() - t;

    let mut all_neighbors = Vec::new();

    // last level
    let id = nearest_neighbors_iter.next().unwrap();

    let neighbors = if let Some(id) = id {
        Some(find_top_neighbors_level(
            id,
            n_neighbors,
            pbwt_trie.div.last().unwrap(),
            &rank_level,
        ))
    } else {
        None
    };

    all_neighbors.push(neighbors);

    //let t = Instant::now();
    for ((trie_level, div_level), id) in pbwt_trie
        .trie
        .iter()
        .rev()
        .zip(
            pbwt_trie
                .div
                .iter()
                .rev()
                .skip(1)
                .chain(std::iter::once(&vec![0])),
        )
        .zip(nearest_neighbors_iter)
    {
        rank_level = build_rank_level(n_neighbors, trie_level, &rank_level);

        let neighbors = if let Some(id) = id {
            Some(find_top_neighbors_level(
                id,
                n_neighbors,
                div_level,
                &rank_level,
            ))
        } else {
            None
        };
        all_neighbors.push(neighbors);
    }
    //let mut _t = NEIGH_T.lock().unwrap();
    //*_t += Instant::now() - t;

    //let t = Instant::now();
    pbwt_input.update(
        last_group_id,
        last_div_above,
        last_div_below,
        pbwt_trie.start_site,
        pbwt_trie.div.last().unwrap(),
        prev_ppa,
        &pbwt_trie.ppa,
    );
    //let mut _t = UPDATE_T.lock().unwrap();
    //*_t += Instant::now() - t;
    all_neighbors
}

use std::rc::Rc;

#[cfg(feature = "obliv")]
pub fn find_top_neighbors_level(
    pos: U16,
    n_neighbors: usize,
    div_level: &[u16],
    nn_ranks: &[Rc<RankList<NNRank>>],
) -> Vec<Usize> {
    let mut result_ranks = RankList::with_elem(n_neighbors, Usize::protect(0));
    for i in 0..nn_ranks.len() as u16 {
        let ranks = find_top_neighbors_level_(i, n_neighbors, div_level, nn_ranks);
        result_ranks.cond_copy_from_slice(&ranks, pos.tp_eq(&i));
    }
    result_ranks.into_iter().collect()
}

#[cfg(not(feature = "obliv"))]
#[inline]
pub fn find_top_neighbors_level(
    pos: u16,
    n_neighbors: usize,
    div_level: &[u16],
    nn_ranks: &[Rc<RankList<NNRank>>],
) -> Vec<Usize> {
    find_top_neighbors_level_(pos, n_neighbors, div_level, nn_ranks)
}

pub fn find_top_neighbors_level_(
    pos: u16,
    n_neighbors: usize,
    div_level: &[u16],
    nn_ranks: &[Rc<RankList<NNRank>>],
) -> Vec<Usize> {
    let pos = pos as usize;
    let mut neighbors = Vec::with_capacity(n_neighbors);
    neighbors.extend(nn_ranks[pos].iter().map(|rank| rank.get_hap_id()));
    if neighbors.len() == n_neighbors {
        return neighbors;
    }

    let mut pos_up_iter = (0..pos).rev();
    let mut pos_down_iter = pos + 1..div_level.len();
    let mut pos_up = pos_up_iter.next();
    let mut pos_down = pos_down_iter.next();

    let mut div_up = 0;
    let mut div_down = 0;

    while neighbors.len() != n_neighbors {
        let mut take_n = n_neighbors - neighbors.len();
        let pos = match (pos_up, pos_down) {
            (Some(pos_up_), Some(pos_down_)) => {
                div_up = div_up.max(div_level[pos_up_ + 1]);
                div_down = div_down.max(div_level[pos_down_]);
                if div_up < div_down {
                    take_n = take_n.min(nn_ranks[pos_up_].len());
                    pos_up = pos_up_iter.next();
                    pos_up_
                } else {
                    take_n = take_n.min(nn_ranks[pos_down_].len());
                    pos_down = pos_down_iter.next();
                    pos_down_
                }
            }
            (Some(pos_up_), None) => {
                take_n = take_n.min(nn_ranks[pos_up_].len());
                pos_up = pos_up_iter.next();
                pos_up_
            }
            (None, Some(pos_down_)) => {
                take_n = take_n.min(nn_ranks[pos_down_].len());
                pos_down = pos_down_iter.next();
                pos_down_
            }
            _ => break,
        };

        neighbors.extend(
            nn_ranks[pos]
                .iter()
                .take(take_n)
                .map(|rank| rank.get_hap_id()),
        );
    }

    neighbors
}
