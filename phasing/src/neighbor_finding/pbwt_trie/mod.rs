mod input;
mod insert;
mod trie;

#[cfg(feature = "obliv")]
mod obliv_insert;

pub use input::PbwtTrieInput;
pub use trie::Node;
pub use trie::PbwtTrie;

use crate::U16;

#[cfg(feature = "obliv")]
use timing_shield::{TpEq, TpOrd};

pub fn nearest_group(group_id: U16, div_above: U16, div_below: U16) -> U16 {
    #[cfg(feature = "obliv")]
    {
        div_above
            .tp_lt_eq(&div_below)
            .select(group_id.tp_eq(&0).select(group_id, group_id - 1), group_id)
    }

    #[cfg(not(feature = "obliv"))]
    if div_above <= div_below {
        if group_id == 0 {
            group_id
        } else {
            group_id - 1
        }
    } else {
        group_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use rand::Rng;

    #[cfg(feature = "obliv")]
    use timing_shield::{TpBool, TpU16};

    #[test]
    fn pbwt_trie() {
        let seed = rand::thread_rng().gen::<u64>();
        println!("seed = {seed}");
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        check_pbwt_trie(&mut rng);
    }

    fn check_pbwt_trie(mut rng: impl Rng) {
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
            .map(|v| TpBool::protect(v))
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

        // expansion test
        {
            let mut test_haps = vec![Vec::<bool>::new(); n_haps];
            for pbwt in &pbwts {
                let expanded = expand_pbwt_trie(pbwt);
                for (i, h) in pbwt
                    .get_index_map(n_haps)
                    .into_iter()
                    .zip(test_haps.iter_mut())
                {
                    h.extend(expanded[i as usize].iter());
                }
            }
            assert_eq!(test_haps, ref_haps);
        }

        // ppa test
        {
            for (block_id, pbwt) in pbwts.iter().enumerate() {
                let mut ref_haps = ref_haps.iter().cloned().enumerate().collect::<Vec<_>>();
                let site_i = (block_id + 1) * n_sites - 1;
                sort_by_reverse_prefix(&mut ref_haps, site_i);
                let (ref_order, _): (Vec<_>, Vec<_>) = ref_haps.into_iter().unzip();
                let test_order = pbwt.ppa.iter().flatten().cloned().collect::<Vec<_>>();
                assert_eq!(ref_order, test_order);
            }
        }

        // div test
        {
            for pbwt in &pbwts {
                let mut expanded: Vec<Vec<bool>> = vec![Vec::with_capacity(0)];
                for (site_i, (layer, div)) in pbwt.trie.iter().zip(pbwt.div.iter()).enumerate() {
                    let mut new_expanded: Vec<Vec<bool>> = vec![Vec::with_capacity(0); div.len()];
                    for (node, h) in layer.into_iter().zip(expanded.iter()) {
                        if let Some(groud_id) = node.0 {
                            let mut h = h.clone();
                            h.push(false);
                            new_expanded[groud_id as usize] = h;
                        }
                        if let Some(groud_id) = node.1 {
                            let mut h = h.clone();
                            h.push(true);
                            new_expanded[groud_id as usize] = h;
                        }
                    }

                    assert_eq!(div[0] as usize, site_i + 1);

                    for (h, &d) in new_expanded.windows(2).zip(div.iter().skip(1)) {
                        let [h1, h2] = h else { panic!() };
                        assert_eq!(compute_div(h1, h2), d as usize);
                    }
                    expanded = new_expanded;
                }
            }
        }

        // input test
        {
            let mut pbwt_input = PbwtTrieInput::new(n_haps);
            let mut prev_ppa = &vec![(0..n_haps).collect::<Vec<_>>()];

            for (block_id, pbwt) in pbwts.iter().enumerate() {
                let expanded = expand_pbwt_trie(pbwt);
                let input_slice = &input[block_id * n_sites..(block_id + 1) * n_sites];

                #[cfg(feature = "obliv")]
                let mut last_groud_id = TpU16::protect(0);
                #[cfg(feature = "obliv")]
                let mut last_d_a = TpU16::protect(0);
                #[cfg(feature = "obliv")]
                let mut last_d_b = TpU16::protect(0);

                #[cfg(not(feature = "obliv"))]
                let mut last_groud_id = 0;
                #[cfg(not(feature = "obliv"))]
                let mut last_d_a = 0;
                #[cfg(not(feature = "obliv"))]
                let mut last_d_b = 0;

                for (i, (groud_id, d_a, d_b)) in
                    pbwt.insert(input_slice.iter().cloned()).enumerate()
                {
                    #[cfg(feature = "obliv")]
                    let input_slice = input_slice
                        .into_iter()
                        .map(|v| v.expose())
                        .collect::<Vec<_>>();

                    let (_, ref_d_a, ref_d_b) = find_input_pos(&input_slice, &expanded, i);

                    // div test
                    #[cfg(not(feature = "obliv"))]
                    assert_eq!((ref_d_a, ref_d_b), (d_a as usize, d_b as usize));

                    #[cfg(feature = "obliv")]
                    assert_eq!(
                        (ref_d_a, ref_d_b),
                        (d_a.expose() as usize, d_b.expose() as usize)
                    );

                    if i == pbwt.n_sites() - 1 {
                        last_groud_id = groud_id;
                        last_d_a = d_a;
                        last_d_b = d_b;
                    }
                }

                pbwt_input.update(
                    last_groud_id,
                    last_d_a,
                    last_d_b,
                    pbwt.start_site,
                    pbwt.div.last().unwrap(),
                    prev_ppa,
                    &pbwt.ppa,
                );

                prev_ppa = &pbwt.ppa;

                #[cfg(feature = "obliv")]
                let input = input.iter().map(|v| v.expose()).collect::<Vec<_>>();

                // ppa test
                let (ref_groud_id, _, _) =
                    find_input_pos(&input, &ref_haps, (block_id + 1) * n_sites - 1);

                #[cfg(not(feature = "obliv"))]
                assert_eq!(ref_groud_id, pbwt_input.full_pos);

                #[cfg(feature = "obliv")]
                assert_eq!(ref_groud_id, pbwt_input.full_pos.expose() as usize);

                // div test
                let ref_div = compute_input_div(&input, &ref_haps, (block_id + 1) * n_sites);

                #[cfg(not(feature = "obliv"))]
                assert_eq!(pbwt_input.full_div, ref_div);

                #[cfg(feature = "obliv")]
                {
                    let full_div = pbwt_input
                        .full_div
                        .iter()
                        .map(|v| v.expose() as usize)
                        .collect::<Vec<_>>();
                    assert_eq!(full_div, ref_div);
                }
            }
        }
    }
}
