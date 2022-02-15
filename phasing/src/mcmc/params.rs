use crate::hmm::{HmmParams, HmmParamsSlice};
use crate::ref_panel::{RefPanel, RefPanelSlice};
use crate::variants::{build_variants, Variant};
use ndarray::{s, Array1, ArrayView1};
use rand::Rng;

const PBWT_MAC: usize = 2;

pub struct McmcSharedParams {
    pub ref_panel: RefPanel,
    pub variants: Array1<Variant>,
    pub hmm_params: HmmParams,
    //pub windows: Vec<(usize, usize)>,
    pub min_window_len_cm: f64,
    pub overlap_region_len: usize,
    pub pbwt_groups: Vec<Vec<usize>>,
    pub s: usize,
}

impl McmcSharedParams {
    pub fn new(
        ref_panel: RefPanel,
        cms: Vec<f64>,
        afreqs: Vec<f64>,
        min_window_len_cm: f64,
        overlap_region_len: usize,
        pbwt_modulo: f64,
        s: usize,
    ) -> Self {
        let variants = build_variants(&cms, &afreqs, ref_panel.n_haps);
        println!("#rare = {}", variants.iter().filter(|v| v.rarity().is_rare()).count());
        let hmm_params = HmmParams::new(&variants, ref_panel.n_haps);
        //let windows = split(&variants, min_window_len_cm);
        let pbwt_groups = Self::pbwt_groups(&variants, pbwt_modulo);
        println!("#pbwt_groups = {}", pbwt_groups.len());
        Self {
            ref_panel,
            variants: Array1::from_vec(variants),
            min_window_len_cm,
            hmm_params,
            //windows,
            overlap_region_len,
            pbwt_groups,
            s,
        }
    }

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> McmcSharedParamsSlice<'a> {
        McmcSharedParamsSlice {
            ref_panel: self.ref_panel.slice(start, end),
            variants: self.variants.slice(s![start..end]),
            hmm_params: self.hmm_params.slice(start, end),
        }
    }

    pub fn randomize_pbwt_bitmask(&self, mut rng: impl Rng) -> Vec<bool> {
        use rand::prelude::SliceRandom;
        let mut pbwt_filter_bitmask = vec![false; self.variants.len()];
        for group in &self.pbwt_groups {
            let i = group.choose(&mut rng).unwrap();
            pbwt_filter_bitmask[*i] = true;
        }
        pbwt_filter_bitmask
    }

    fn pbwt_groups(variants: &[Variant], pbwt_modulo: f64) -> Vec<Vec<usize>> {
        let mut start_cm = variants[0].cm;
        let mut end_cm = start_cm + pbwt_modulo;
        let mut pbwt_groups = Vec::new();

        let mut cur_group = Vec::new();
        for (i, variant) in variants.iter().enumerate() {
            if variant.cm >= start_cm && variant.cm < end_cm {
                if variant.get_mac() >= PBWT_MAC {
                    cur_group.push(i);
                }
            } else {
                if !cur_group.is_empty() {
                    pbwt_groups.push(cur_group);
                    cur_group = Vec::new();
                }
                while variant.cm >= end_cm {
                    start_cm += pbwt_modulo;
                    end_cm += pbwt_modulo;
                }
            }
        }
        pbwt_groups
    }
}

pub struct McmcSharedParamsSlice<'a> {
    pub ref_panel: RefPanelSlice<'a>,
    pub variants: ArrayView1<'a, Variant>,
    pub hmm_params: HmmParamsSlice<'a>,
}
