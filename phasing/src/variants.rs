use crate::Genotype;
use ndarray::ArrayView1;

const RARE_VARIANT_FREQ: f64 = 0.001;

#[derive(Clone, Copy, PartialEq)]
pub enum Rarity {
    NotRare,
    Rare(bool), // is Ref AF > 0.5 (is Ref major allele)?
}

impl Rarity {
    pub fn is_rare(self) -> bool {
        self != Self::NotRare
    }
}

#[derive(Debug)]
pub struct Variant {
    pub bp: u32,
    pub cm: f64,
    afreq: f64,
    cref: usize,
    calt: usize,
}

impl Variant {
    pub fn new(genotype_calt: usize, bp: u32, cm: f64, afreq: f64, n_haps: usize) -> Self {
        let calt = (n_haps as f64 * afreq).round() as usize + genotype_calt;
        let cref = n_haps - calt + 2;
        let afreq = calt as f64 / (calt + cref) as f64;
        Self {
            bp,
            cm,
            afreq,
            cref,
            calt,
        }
    }

    pub fn get_mac(&self) -> usize {
        self.cref.min(self.calt)
    }

    pub fn get_maf(&self) -> f64 {
        self.afreq.min(1. - self.afreq)
    }

    pub fn get_ref_af(&self) -> f64 {
        1. - self.afreq
    }

    pub fn rarity(&self) -> Rarity {
        if self.get_maf() <= RARE_VARIANT_FREQ {
            Rarity::Rare(self.get_ref_af() > 0.5)
        } else {
            Rarity::NotRare
        }
    }
}

pub fn build_variants(
    genotypes: ArrayView1<Genotype>,
    bps: &[u32],
    cms: &[f64],
    afreqs: &[f64],
    n_haps: usize,
) -> Vec<Variant> {
    genotypes
        .iter()
        .zip(bps.iter())
        .zip(afreqs.iter())
        .zip(cms.iter())
        .map(|(((&g, &bp), &afreq), &cm)| {
            Variant::new(if g != -1 { g as usize } else { 0 }, bp, cm, afreq, n_haps)
        })
        .collect()
}
