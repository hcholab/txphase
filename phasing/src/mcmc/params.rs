use crate::hmm::HmmParams;
use crate::variants::{build_variants, Variant};
use common::ref_panel::{RefPanel, RefPanelSlice};
#[cfg(feature = "obliv")]
use compressed_pbwt_obliv::pbwt_trie::PbwtTrie;

#[cfg(not(feature = "obliv"))]
use compressed_pbwt::pbwt_trie::PbwtTrie;
use ndarray::{s, Array1, ArrayView1};
use rand::Rng;

const PBWT_MAC: usize = 2;

pub struct McmcSharedParams {
    pub ref_panel: RefPanel,
    pub variants: Array1<Variant>,
    pub hmm_params: HmmParams,
    pub min_window_len_cm: f64,
    pub overlap_region_len: usize,
    pub pbwt_tries: Vec<PbwtTrie>,
    pub pbwt_evaluted: Vec<bool>,
    pub pbwt_groups: Vec<Vec<usize>>,
    pub s: usize,
}

impl McmcSharedParams {
    pub fn new(
        ref_panel: RefPanel,
        bps: Vec<u32>,
        cms: Vec<f64>,
        afreqs: Vec<f64>,
        min_window_len_cm: f64,
        overlap_region_len: usize,
        pbwt_modulo: f64,
        s: usize,
    ) -> Self {
        let variants = build_variants(&bps, &cms, &afreqs, ref_panel.n_haps);
        println!(
            "#rare = {}",
            variants.iter().filter(|v| v.rarity().is_rare()).count()
        );
        let hmm_params = HmmParams::new(ref_panel.n_haps);
        let (pbwt_evaluted, pbwt_groups) = Self::get_pbwt_evaluted(&variants, pbwt_modulo);

        let mut pbwt_tries = Vec::<PbwtTrie>::new();

        ref_panel
            .blocks
            .iter()
            .zip(std::iter::once(0).chain(ref_panel.block_map.iter().cloned()))
            .for_each(|(block, start_site)| {
                let block = block.as_slice();
                let index_map = block
                    .index_map
                    .iter()
                    .map(|&v| v as u16)
                    .collect::<Vec<_>>();
                let pbwt = if let Some(prev_pbwt) = pbwt_tries.last() {
                    PbwtTrie::transform(
                        start_site,
                        block.iter(),
                        &prev_pbwt.ppa,
                        &index_map,
                        block.n_unique(),
                        block.n_sites(),
                    )
                } else {
                    let ppa = vec![(0..ref_panel.n_haps).collect::<Vec<_>>()];
                    PbwtTrie::transform(
                        start_site,
                        block.iter(),
                        &ppa,
                        &index_map,
                        block.n_unique(),
                        block.n_sites(),
                    )
                };
                pbwt_tries.push(pbwt);
            });

        println!("#pbwt_groups = {}", pbwt_groups.len());
        Self {
            ref_panel,
            variants: Array1::from_vec(variants),
            min_window_len_cm,
            hmm_params,
            overlap_region_len,
            pbwt_tries,
            pbwt_evaluted,
            pbwt_groups,
            s,
        }
    }

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> McmcSharedParamsSlice<'a> {
        McmcSharedParamsSlice {
            ref_panel: self.ref_panel.slice(start, end),
            variants: self.variants.slice(s![start..end]),
        }
    }

    pub fn randomize_pbwt_group_bitmask(&self, mut rng: impl Rng) -> Vec<bool> {
        use rand::prelude::SliceRandom;
        let mut filter = vec![false; self.variants.len()];
        for group in &self.pbwt_groups {
            let i = group.choose(&mut rng).unwrap();
            filter[*i] = true;
        }
        filter
    }

    fn get_pbwt_evaluted(variants: &[Variant], pbwt_modulo: f64) -> (Vec<bool>, Vec<Vec<usize>>) {
        let mut pbwt_evaluted = vec![true; variants.len()];
        let mut last_group_id = 0;
        let mut pbwt_groups = Vec::new();
        let mut cur_group = Vec::new();
        for (i, v) in variants.iter().enumerate() {
            if v.get_mac() < PBWT_MAC {
                pbwt_evaluted[i] = false;
                continue;
            }
            let group_id = (v.cm / pbwt_modulo).round() as usize;
            if group_id != last_group_id {
                if !cur_group.is_empty() {
                    pbwt_groups.push(cur_group);
                    cur_group = Vec::new();
                }
                last_group_id = group_id;
            }
            cur_group.push(i);
        }
        (pbwt_evaluted, pbwt_groups)
    }
}

pub struct McmcSharedParamsSlice<'a> {
    pub ref_panel: RefPanelSlice<'a>,
    pub variants: ArrayView1<'a, Variant>,
}
