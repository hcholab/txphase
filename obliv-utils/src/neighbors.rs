use crate::obliv::top_s::RankShort;

use timing_shield::{TpEq, TpU16, TpU64};

pub(crate) fn find_top_neighbors_layer(
    node_id: TpU16,
    n_neighbors: usize,
    div_layer: &[u16],
    neighbor_layer: &[RankShort],
) -> Vec<TpU64> {
    let mut neighbors = vec![TpU64::protect(0); n_neighbors];
    for id in 0..neighbor_layer.len() as u16 {
        let cond = node_id.tp_eq(&id);
        let mut new_neighbors =
            crate::pbwt_trie::find_top_neighbors_layer(id, n_neighbors, div_layer, neighbor_layer);

        cond.cond_swap(&mut neighbors[..], &mut new_neighbors[..]);

        if new_neighbors.len() < neighbors.len() {
            let first = *neighbors.first().unwrap();
            for i in &mut neighbors[new_neighbors.len()..] {
                *i = first;
            }
        }
    }
    neighbors
}
