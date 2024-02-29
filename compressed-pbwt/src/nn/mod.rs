mod nn;
pub mod nn_tree;

pub use nn::*;
pub use nn_tree::*;

#[cfg(not(feature = "obliv"))]
mod rank;

#[cfg(feature = "obliv")]
mod obliv_rank;

#[cfg(feature = "obliv")]
use obliv_rank::NNRank;

#[cfg(not(feature = "obliv"))]
use rank::NNRank;

#[cfg(feature = "obliv")]
pub type RankList<T> = obliv_utils::vec::OblivVec<T>;

#[cfg(not(feature = "obliv"))]
pub type RankList<T> = Vec<T>;

pub use nn::find_top_neighbors;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbwt_trie::PbwtTrie;
    use crate::test_utils::*;
    use rand::Rng;

    #[test]
    fn nearest_neighbors() {
        let seed = rand::thread_rng().gen::<u64>();
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let n_unique = 10;
        let multiplier = 2;
        let n_haps = n_unique * multiplier;
        let n_sites = 20;
        let n_blocks = 30;
        let blocks = (0..n_blocks)
            .map(|_| gen_unique_haps_block(n_unique, n_sites, &mut rng))
            .collect::<Vec<_>>();

        let index_maps = (0..n_blocks)
            .map(|_| gen_index_map(n_unique, multiplier, &mut rng))
            .collect::<Vec<_>>();

        let n_sites_all = vec![n_sites; n_blocks];

        let ref_haps = expand_ref(&blocks, &n_sites_all, &index_maps);

        let mut input = vec![false; n_sites * n_blocks];
        rng.fill(&mut input[..]);

        #[cfg(feature = "obliv")]
        let input = input
            .into_iter()
            .map(|v| timing_shield::TpBool::protect(v))
            .collect::<Vec<_>>();

        let mut prev_ppa = &vec![(0..n_haps).collect::<Vec<_>>()];
        let mut pbwts = Vec::new();
        for (i, (block, index_map)) in blocks.iter().zip(index_maps.iter()).enumerate() {
            let pbwt = PbwtTrie::transform(
                i * n_sites,
                hap_iter(block, n_sites),
                prev_ppa,
                index_map,
                n_unique,
                n_sites,
            );
            pbwts.push(pbwt);
            prev_ppa = &pbwts.last().unwrap().ppa;
        }

        let test_nn =
            nn::find_top_neighbors(&input, n_haps, &pbwts, n_haps, &vec![true; input.len()]);

        #[cfg(feature = "obliv")]
        let input = input.into_iter().map(|v| v.expose()).collect::<Vec<_>>();

        #[cfg(feature = "obliv")]
        let test_nn = test_nn
            .into_iter()
            .map(|v| {
                v.unwrap()
                    .into_iter()
                    .map(|v| v.expose() as usize)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        #[cfg(not(feature = "obliv"))]
        let test_nn = test_nn
            .into_iter()
            .map(|v| v.unwrap().into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();

        for (i, nn) in test_nn.into_iter().enumerate() {
            let input_div = compute_input_div(&input, &ref_haps, i + 1);
            let test_div = nn.into_iter().map(|i| input_div[i]).collect::<Vec<_>>();
            assert!(test_div.is_sorted());
        }
    }
}
