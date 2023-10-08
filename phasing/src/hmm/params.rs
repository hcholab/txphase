use crate::genotype_graph::GenotypeGraph;
use crate::variants::Variant;
use crate::{tp_value, tp_value_real, Bool, Real};
use ndarray::ArrayView1;
#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::{TpEq, TpOrd, TpU32};

const EPROB: f64 = 0.0001;
const N_EFF: f64 = 15000.;
const MIN_DIST: f64 = 1e-7;
const R_CONST: f64 = 0.04 * N_EFF;

pub struct HmmParams {
    pub eprob: Real,
    n_haps_ref_frac: f64,
    r_const: f64,
    #[cfg(feature = "obliv")]
    n_haps_ref_frac_obliv: Real,
    #[cfg(feature = "obliv")]
    r_const_obliv: Real,
}

impl HmmParams {
    pub fn new(n_haps_ref: usize) -> Self {
        let n_haps_ref_frac = 1. / n_haps_ref as f64;
        let r_const = R_CONST / n_haps_ref as f64;
        Self {
            eprob: tp_value_real!(EPROB / (1. - EPROB), f32),
            n_haps_ref_frac,
            r_const,
            #[cfg(feature = "obliv")]
            n_haps_ref_frac_obliv: tp_value_real!(n_haps_ref_frac, f32),
            #[cfg(feature = "obliv")]
            r_const_obliv: tp_value_real!(r_const, f32),
        }
    }

    pub fn get_rprobs(
        &self,
        ignored_sites: ArrayView1<Bool>,
        genotype_graph: &GenotypeGraph,
        variants: ArrayView1<Variant>,
    ) -> Rprobs {
        let mut fwd_rprobs = Vec::new();
        let mut bwd_rprobs = Vec::new();

        let mut n_skips = tp_value!(0, u32);

        let mut last_cm = tp_value_real!(variants[0].cm, f32);

        for i in 1..variants.len() {
            #[cfg(feature = "obliv")]
            let r = {
                let cond = !ignored_sites[i];
                let r0 = {
                    let cm = variants[i].cm - variants[i - 1].cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                };

                let r1 = {
                    if i >= 2 {
                        let cm = variants[i].cm - variants[i - 2].cm;
                        compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                    } else {
                        (tp_value_real!(0, i64), tp_value_real!(0, i64))
                    }
                };

                let r2 = {
                    let cm = tp_value_real!(variants[i].cm, f32) - last_cm;
                    compute_recomb_prob_obliv(cm, self.r_const_obliv, self.n_haps_ref_frac_obliv)
                };

                let r = (
                    cond.select(
                        n_skips
                            .tp_eq(&TpU32::protect(0))
                            .select(r0.0, n_skips.tp_eq(&TpU32::protect(1)).select(r1.0, r2.0)),
                        tp_value_real!(0, i64),
                    ),
                    cond.select(
                        n_skips
                            .tp_eq(&TpU32::protect(0))
                            .select(r0.1, n_skips.tp_eq(&TpU32::protect(1)).select(r1.1, r2.1)),
                        tp_value_real!(0, i64),
                    ),
                );

                last_cm = cond.select(Real::protect_f32(variants[i].cm as f32), last_cm);
                n_skips = cond.select(TpU32::protect(0), n_skips + 1);

                r
            };

            #[cfg(not(feature = "obliv"))]
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
                (0., 0.)
            };

            fwd_rprobs.push(r);
        }

        n_skips = tp_value!(0, u32);

        last_cm = tp_value_real!(variants.last().unwrap().cm, f32);

        for i in (0..variants.len() - 1).rev() {
            #[cfg(feature = "obliv")]
            let r = {
                let cond = !ignored_sites[i] | genotype_graph.graph[i + 1].is_segment_marker();
                let r0 = {
                    let cm = variants[i + 1].cm - variants[i].cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                };
                let r1 = {
                    if i + 2 <= variants.len() - 1 {
                        let cm = variants[i + 2].cm - variants[i].cm;
                        compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                    } else {
                        (tp_value_real!(0, i64), tp_value_real!(0, i64))
                    }
                };

                let r2 = {
                    let cm = last_cm - tp_value_real!(variants[i].cm, f32);
                    compute_recomb_prob_obliv(cm, self.r_const_obliv, self.n_haps_ref_frac_obliv)
                };

                let r = (
                    cond.select(
                        n_skips
                            .tp_eq(&TpU32::protect(0))
                            .select(r0.0, n_skips.tp_eq(&TpU32::protect(1)).select(r1.0, r2.0)),
                        tp_value_real!(0, i64),
                    ),
                    cond.select(
                        n_skips
                            .tp_eq(&TpU32::protect(0))
                            .select(r0.1, n_skips.tp_eq(&TpU32::protect(1)).select(r1.1, r2.1)),
                        tp_value_real!(0, i64),
                    ),
                );

                last_cm = cond.select(Real::protect_f32(variants[i].cm as f32), last_cm);
                n_skips = cond.select(TpU32::protect(0), n_skips + 1);
                r
            };

            #[cfg(not(feature = "obliv"))]
            let r = if !ignored_sites[i] || genotype_graph.graph[i + 1].is_segment_marker() {
                let r = if n_skips == 0 {
                    let cm = variants[i + 1].cm - variants[i].cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                } else if n_skips == 1 {
                    let cm = variants[i + 2].cm - variants[i].cm;
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                } else {
                    let cm = last_cm - tp_value_real!(variants[i].cm, f32);
                    compute_recomb_prob(cm, self.r_const, self.n_haps_ref_frac)
                };

                last_cm = tp_value_real!(variants[i].cm, f32);
                n_skips = 0;
                r
            } else {
                n_skips += 1;
                (tp_value_real!(0, i64), tp_value_real!(0, i64))
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
    fwd: Vec<(Real, Real)>,
    bwd: Vec<(Real, Real)>,
}

impl Rprobs {
    pub fn slice<'a>(&'a self, start: usize, end: usize) -> RprobsSlice<'a> {
        RprobsSlice {
            fwd: &self.fwd[start..end - 1],
            bwd: &self.bwd[start..end - 1],
        }
    }
}

pub struct RprobsSlice<'a> {
    fwd: &'a [(Real, Real)],
    bwd: &'a [(Real, Real)],
}

impl<'a> RprobsSlice<'a> {
    pub fn get_forward(&self) -> Box<dyn Iterator<Item = (Real, Real)>> {
        Box::new(self.fwd.to_owned().into_iter())
    }

    pub fn get_backward(&self) -> Box<dyn Iterator<Item = (Real, Real)>> {
        let mut bwd = self.bwd.to_owned();
        bwd.reverse();
        Box::new(bwd.into_iter())
    }
}

#[inline]
fn compute_recomb_prob(dist_cm: f64, r_const: f64, n_haps_ref_frac: f64) -> (Real, Real) {
    let r = -(-r_const * dist_cm.max(MIN_DIST)).exp_m1();
    (
        tp_value_real!(r * n_haps_ref_frac, f32),
        tp_value_real!(1. - r, f32),
    )
}

#[inline]
#[cfg(feature = "obliv")]
fn compute_recomb_prob_obliv(
    dist_cm: Real,
    r_const: Real,
    n_haps_ref_frac: Real,
) -> (Real, Real) {
    let dist_cm = {
        let min_dist = Real::protect_f32(MIN_DIST as f32);
        dist_cm.tp_gt(&min_dist).select(dist_cm, min_dist)
    };
    let x = r_const * dist_cm;
    let r = x.tp_lt(&Real::protect_f32(0.02)).select(x, x.ode());
    (r * n_haps_ref_frac, Real::protect_i64(1) - r)
}
