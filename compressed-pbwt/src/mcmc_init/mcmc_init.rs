use crate::mcmc_init::init_single_site::init_single_site;
use crate::mcmc_init::init_tree::build_init_tree;
use crate::pbwt_trie::{PbwtTrie, PbwtTrieInput};
use crate::{Bool, I8};

#[cfg(feature = "obliv")]
use crate::U16;

#[cfg(feature = "obliv")]
use timing_shield::TpEq;

pub fn mcmc_init(
    genotypes: &[I8],
    pbwt_tries: &[PbwtTrie],
    n_haps: usize,
) -> (Vec<Bool>, Vec<Bool>) {
    let mut pbwt_trie_input_0 = PbwtTrieInput::new(n_haps);
    let mut pbwt_trie_input_1 = PbwtTrieInput::new(n_haps);

    let mut h_0 = Vec::with_capacity(genotypes.len());
    let mut h_1 = Vec::with_capacity(genotypes.len());

    #[cfg(feature = "obliv")]
    {
        let g = genotypes[0] | (genotypes[0] >> 1);
        h_0.push((g & 1).tp_eq(&1));
        h_1.push(((g >> 1) & 1).tp_eq(&1));
    }

    #[cfg(not(feature = "obliv"))]
    match genotypes[0] {
        1 => (h_0.push(true), h_1.push(false)),
        0 => (h_0.push(false), h_1.push(false)),
        2 => (h_0.push(true), h_1.push(true)),
        _ => panic!("This shouldn't happen"),
    };

    for (cur_pbwt, prev_ppa, next_first_full_haps) in iter_pbwt_tries(pbwt_tries, n_haps) {
        let init_tree_0 = build_init_tree(
            &cur_pbwt.trie,
            &pbwt_trie_input_0,
            prev_ppa,
            &cur_pbwt.ppa,
            next_first_full_haps.as_deref(),
        );

        let init_tree_1 = build_init_tree(
            &cur_pbwt.trie,
            &pbwt_trie_input_1,
            prev_ppa,
            &cur_pbwt.ppa,
            next_first_full_haps.as_deref(),
        );

        let start_site = cur_pbwt.start_site + 1;
        let genotypes_slice =
            &genotypes[start_site..(start_site + cur_pbwt.n_sites()).min(genotypes.len())];

        let (sender_0, receiver_0) = std::sync::mpsc::channel::<Bool>();
        let (sender_1, receiver_1) = std::sync::mpsc::channel::<Bool>();

        sender_0.send(*h_0.last().unwrap()).unwrap();
        sender_1.send(*h_1.last().unwrap()).unwrap();

        let neighbors_iter_0 = cur_pbwt.insert(receiver_0.into_iter());
        let neighbors_iter_1 = cur_pbwt.insert(receiver_1.into_iter());
        #[cfg(feature = "obliv")]
        let (mut last_id_0, mut last_d_a_0, mut last_d_b_0) =
            (U16::protect(0), U16::protect(0), U16::protect(0));
        #[cfg(feature = "obliv")]
        let (mut last_id_1, mut last_d_a_1, mut last_d_b_1) =
            (U16::protect(0), U16::protect(0), U16::protect(0));

        #[cfg(not(feature = "obliv"))]
        let (mut last_id_0, mut last_d_a_0, mut last_d_b_0) = (0, 0, 0);
        #[cfg(not(feature = "obliv"))]
        let (mut last_id_1, mut last_d_a_1, mut last_d_b_1) = (0, 0, 0);

        for (
            i,
            (
                (
                    ((init_level_0, init_level_1), ((id_0, d_a_0, d_b_0), (id_1, d_a_1, d_b_1))),
                    div_level,
                ),
                g,
            ),
        ) in init_tree_0
            .into_iter()
            .zip(init_tree_1.into_iter())
            .zip(neighbors_iter_0.zip(neighbors_iter_1))
            .zip(cur_pbwt.div.iter())
            .zip(genotypes_slice)
            .enumerate()
        {
            #[cfg(feature = "obliv")]
            let (next_h_0, next_h_1) = {
                let site_i = i + cur_pbwt.start_site;
                let phase = init_single_site(
                    cur_pbwt.start_site,
                    site_i,
                    id_0,
                    d_a_0,
                    d_b_0,
                    id_1,
                    d_a_1,
                    d_b_1,
                    div_level,
                    &init_level_0,
                    &init_level_1,
                );

                let next_h_0 = g.tp_eq(&1).select(phase.0, (*g >> 1).tp_eq(&1));
                let next_h_1 = g.tp_eq(&1).select(phase.1, (*g >> 1).tp_eq(&1));
                (next_h_0, next_h_1)
            };

            #[cfg(not(feature = "obliv"))]
            let (next_h_0, next_h_1) = match g {
                1 => {
                    let site_i = i + cur_pbwt.start_site;
                    init_single_site(
                        cur_pbwt.start_site,
                        site_i,
                        id_0,
                        d_a_0,
                        d_b_0,
                        id_1,
                        d_a_1,
                        d_b_1,
                        div_level,
                        &init_level_0,
                        &init_level_1,
                    )
                }
                0 => (false, false),
                2 => (true, true),
                _ => panic!("This should never happen."),
            };

            h_0.push(next_h_0);
            h_1.push(next_h_1);

            if i == cur_pbwt.n_sites() - 1 {
                (last_id_0, last_d_a_0, last_d_b_0) = (id_0, d_a_0, d_b_0);
                (last_id_1, last_d_a_1, last_d_b_1) = (id_1, d_a_1, d_b_1);
            } else {
                sender_0.send(next_h_0).unwrap();
                sender_1.send(next_h_1).unwrap();
            }
        }
        let zero_ppa;
        let prev_ppa = prev_ppa.unwrap_or({
            zero_ppa = vec![(0..n_haps as u32).collect::<Vec<_>>()];
            &zero_ppa
        });

        pbwt_trie_input_0.update(
            last_id_0,
            last_d_a_0,
            last_d_b_0,
            cur_pbwt.start_site,
            cur_pbwt.div.last().unwrap(),
            prev_ppa,
            &cur_pbwt.ppa,
        );

        pbwt_trie_input_1.update(
            last_id_1,
            last_d_a_1,
            last_d_b_1,
            cur_pbwt.start_site,
            cur_pbwt.div.last().unwrap(),
            prev_ppa,
            &cur_pbwt.ppa,
        );
    }
    (h_0, h_1)
}

pub fn iter_pbwt_tries(
    pbwt_tries: &[PbwtTrie],
    n_haps: usize,
) -> impl Iterator<Item = (&PbwtTrie, Option<&[Vec<u32>]>, Option<Vec<bool>>)> + '_ {
    use std::iter::once;
    pbwt_tries
        .iter()
        .zip(pbwt_tries.iter().skip(1).map(|v| Some(v)).chain(once(None)))
        .zip(once(None).chain(pbwt_tries.iter().map(|v| Some(v))))
        .map(move |((cur_pbwt, next_pbwt), prev_pbwt)| {
            let prev_ppa = prev_pbwt.map(|v| &v.ppa[..]);
            let next_first_full_haps = next_pbwt.map(|v| {
                v.get_index_map(n_haps)
                    .into_iter()
                    .map(|hap_id| v.first_haps[hap_id as usize])
                    .collect::<Vec<_>>()
            });

            (cur_pbwt, prev_ppa, next_first_full_haps)
        })
}
