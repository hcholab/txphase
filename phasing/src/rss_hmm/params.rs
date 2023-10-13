use crate::genotype_graph::GenotypeGraph;
use crate::variants::Variant;
use ndarray::ArrayView1;

use crate::inner::*;

const EPROB: f64 = 0.0001;
const N_EFF: f64 = 15000.;
const MIN_DIST: f64 = 1e-7;
const R_CONST: f64 = 0.04 * N_EFF;

pub struct HmmParams {
    pub eprob: Real,
    n_haps_ref_frac: f64,
    r_const: f64,
}

impl HmmParams {
    pub fn new(n_haps_ref: usize) -> Self {
        let n_haps_ref_frac = 1. / n_haps_ref as f64;
        let r_const = R_CONST / n_haps_ref as f64;
        let eprob = EPROB / (1. - EPROB);
        #[cfg(feature = "obliv")]
        let eprob = Real::protect_f32(eprob as f32);
        Self {
            eprob,
            n_haps_ref_frac,
            r_const,
        }
    }

    pub fn get_rprobs(
        &self,
        ignored_sites: ArrayView1<bool>,
        genotype_graph: &GenotypeGraph,
        variants: ArrayView1<Variant>,
    ) -> Rprobs {
        let mut fwd_rprobs = Vec::new();
        let mut bwd_rprobs = Vec::new();

        let mut n_skips = 0;

        let mut last_cm = variants[0].cm;

        for i in 1..variants.len() {
            let r = if !ignored_sites[i] {
                let r = if n_skips == 0 {
                    let cm = variants[i].cm - variants[i - 1].cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                } else if n_skips == 1 {
                    let cm = variants[i].cm - variants[i - 2].cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                } else {
                    let cm = variants[i].cm - last_cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                };

                last_cm = variants[i].cm;
                n_skips = 0;
                r
            } else {
                n_skips += 1;
                0.
            };

            fwd_rprobs.push(r);
        }

        n_skips = 0;

        last_cm = variants.last().unwrap().cm;

        for i in (0..variants.len() - 1).rev() {
            let r = if !ignored_sites[i] || genotype_graph.graph[i + 1].is_segment_marker() {
                let r = if n_skips == 0 {
                    let cm = variants[i + 1].cm - variants[i].cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                } else if n_skips == 1 {
                    let cm = variants[i + 2].cm - variants[i].cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                } else {
                    let cm = last_cm - variants[i].cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                };

                last_cm = variants[i].cm;
                n_skips = 0;
                r
            } else {
                n_skips += 1;
                0.
            };

            bwd_rprobs.push(r);
        }

        bwd_rprobs.reverse();
        Rprobs {
            fwd: fwd_rprobs,
            bwd: bwd_rprobs,
        }
    }
}

pub struct Rprobs {
    fwd: Vec<f64>,
    bwd: Vec<f64>,
}

impl Rprobs {
    pub fn slice<'a>(&'a self, start: usize, end: usize) -> RprobsSlice<'a> {
        RprobsSlice {
            fwd: &self.fwd[start..end - 1],
            bwd: &self.bwd[start..end - 1],
        }
    }
    pub fn as_slice<'a>(&'a self) -> RprobsSlice<'a> {
        RprobsSlice {
            fwd: &self.fwd.as_slice(),
            bwd: &self.bwd.as_slice(),
        }
    }
}

pub struct RprobsSlice<'a> {
    fwd: &'a [f64],
    bwd: &'a [f64],
}

impl<'a> RprobsSlice<'a> {
    pub fn get_forward(&self) -> Box<dyn Iterator<Item = f64>> {
        Box::new(self.fwd.to_owned().into_iter())
    }

    pub fn get_backward(&self) -> Box<dyn Iterator<Item = f64>> {
        let mut bwd = self.bwd.to_owned();
        bwd.reverse();
        Box::new(bwd.into_iter())
    }
}

#[inline]
fn compute_recomb_prob(dist_cm: f64, r_const: f64, n_haps_ref_frac: f64) -> f64 {
    let r = -(-r_const * dist_cm.max(MIN_DIST)).exp_m1();
    r * n_haps_ref_frac / (1. - r)
}
