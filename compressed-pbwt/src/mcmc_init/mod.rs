mod init_single_site;
mod init_tree;
mod mcmc_init;

#[cfg(not(feature = "obliv"))]
mod rank;

#[cfg(feature = "obliv")]
mod obliv_rank;

#[cfg(feature = "obliv")]
use obliv_rank::InitRank;

#[cfg(not(feature = "obliv"))]
use rank::InitRank;

#[cfg(feature = "obliv")]
pub type RankList<T> = obliv_utils::vec::OblivVec<T>;

#[cfg(not(feature = "obliv"))]
pub type RankList<T> = Vec<T>;

pub use mcmc_init::mcmc_init;

//#[cfg(test)]
//mod tests {
//use super::*;
//use crate::mcmc_init::init_single_site::process_top_init_neighbors;
//use crate::pbwt_trie::{PbwtTrie, PbwtTrieInput};
//use crate::test_utils::*;
//use rand::Rng;

//#[cfg(feature = "obliv")]
//use timing_shield::{TpBool, TpU16};

//#[test]
//fn mcmc_init() {
//let seed = rand::thread_rng().gen::<u64>();
////let seed = 17010676330651928772;
//println!("seed = {seed}");
//use rand::SeedableRng;
//let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

//let n_unique = 10;
//let multiplier = 2;
//let n_haps = n_unique * multiplier;
//let n_sites = 20;
//let n_blocks = 30;
//let blocks = (0..n_blocks)
//.map(|_| gen_unique_haps_block(n_unique, n_sites, &mut rng))
//.collect::<Vec<_>>();

//let index_maps = (0..n_blocks)
//.map(|_| gen_index_map(n_unique, multiplier, &mut rng))
//.collect::<Vec<_>>();

//let n_sites_all = vec![n_sites; n_blocks];

//let ref_haps = expand_ref(&blocks, &n_sites_all, &index_maps);

//let mut input = vec![false; n_sites * n_blocks];
//rng.fill(&mut input[..]);

//#[cfg(feature = "obliv")]
//let plaintext_input = input;

//#[cfg(feature = "obliv")]
//let input = plaintext_input
//.iter()
//.map(|&v| TpBool::protect(v))
//.collect::<Vec<_>>();

//let mut prev_ppa = &vec![(0..n_haps).collect::<Vec<_>>()];
//let mut pbwts = Vec::new();
//for (i, (block, index_map)) in blocks.iter().zip(index_maps.iter()).enumerate() {
//let pbwt = PbwtTrie::transform(
//i * n_sites,
//hap_iter(block, n_sites),
//prev_ppa,
//index_map,
//n_unique,
//n_sites,
//);
//pbwts.push(pbwt);
//prev_ppa = &pbwts.last().unwrap().ppa;
//}

//// init tree
//{
//let mut pbwt_input = PbwtTrieInput::new(n_haps);

//for (block_id, (cur_pbwt, prev_ppa, next_first_full_haps)) in
//mcmc_init::iter_pbwt_tries(&pbwts, n_haps).enumerate()
//{
//let init_tree = init_tree::build_init_tree(
//&cur_pbwt.trie,
//&pbwt_input,
//prev_ppa,
//&cur_pbwt.ppa,
//next_first_full_haps.as_deref(),
//);

//let input_slice =
//&input[block_id * n_sites..((block_id + 1) * n_sites).min(input.len())];

//#[cfg(feature = "obliv")]
//let (mut last_groud_id, mut last_d_a, mut last_d_b) =
//(TpU16::protect(0), TpU16::protect(0), TpU16::protect(0));

//#[cfg(not(feature = "obliv"))]
//let (mut last_groud_id, mut last_d_a, mut last_d_b) = (0, 0, 0);

//for (i, (((groud_id, d_a, d_b), init), div)) in cur_pbwt
//.insert(input_slice.iter().cloned())
//.zip(init_tree.iter())
//.zip(cur_pbwt.div.iter())
//.enumerate()
//{
//let site_i = block_id * n_sites + i;
//for rank in init.iter().map(|v| v.iter()).flatten() {
//#[cfg(feature = "obliv")]
//let ref_div = compute_div(
//&ref_haps[rank.get_hap_id().expose() as usize][..block_id * n_sites],
//&plaintext_input[..block_id * n_sites],
//);

//#[cfg(not(feature = "obliv"))]
//let ref_div = compute_div(
//&ref_haps[rank.get_hap_id()][..block_id * n_sites],
//&input[..block_id * n_sites],
//);

//#[cfg(feature = "obliv")]
//assert_eq!(ref_div, rank.get_div().expose() as usize);

//#[cfg(not(feature = "obliv"))]
//assert_eq!(ref_div, rank.get_div());

//if site_i + 1 < input.len() {
//#[cfg(feature = "obliv")]
//assert_eq!(
//ref_haps[rank.get_hap_id().expose() as usize][site_i + 1],
//rank.get_hap().expose()
//);

//#[cfg(not(feature = "obliv"))]
//assert_eq!(ref_haps[rank.get_hap_id()][site_i + 1], rank.get_hap());
//}
//}

//let ranks = process_top_init_neighbors(
//block_id * n_sites,
//site_i,
//groud_id,
//d_a,
//d_b,
//div,
//init,
//);

//for rank in ranks {
//#[cfg(feature = "obliv")]
//let ref_match = compute_rev_prefix_match(
//&ref_haps[rank.2.expose() as usize][..site_i + 1],
//&plaintext_input[..site_i + 1],
//);

//#[cfg(not(feature = "obliv"))]
//let ref_match = compute_rev_prefix_match(
//&ref_haps[rank.2][..site_i + 1],
//&input[..site_i + 1],
//);

//#[cfg(feature = "obliv")]
//assert_eq!(ref_match, rank.0.expose() as usize);

//#[cfg(not(feature = "obliv"))]
//assert_eq!(ref_match, rank.0);

//if site_i + 1 < input.len() {
//#[cfg(feature = "obliv")]
//assert_eq!(
//ref_haps[rank.2.expose() as usize][site_i + 1],
//rank.1.expose()
//);

//#[cfg(not(feature = "obliv"))]
//assert_eq!(ref_haps[rank.2][site_i + 1], rank.1);
//}
//}

//if i == cur_pbwt.n_sites() - 1 {
//last_groud_id = groud_id;
//last_d_a = d_a;
//last_d_b = d_b;
//}
//}

//let zero_ppa;
//let prev_ppa = prev_ppa.unwrap_or({
//zero_ppa = vec![(0..n_haps).collect::<Vec<_>>()];
//&zero_ppa
//});

//pbwt_input.update(
//last_groud_id,
//last_d_a,
//last_d_b,
//cur_pbwt.start_site,
//cur_pbwt.div.last().unwrap(),
//prev_ppa,
//&cur_pbwt.ppa,
//);
//}
//}
//}
//}
