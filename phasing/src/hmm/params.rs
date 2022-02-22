use crate::variants::Variant;
use crate::Real;

const EPROB: f64 = 0.0001;
const N_EFF: usize = 15000;

pub struct HmmParams {
    pub eprob: (Real, Real),           // (e, 1 - e)
    forward_rprobs: Vec<(Real, Real)>, //(r, 1 - r)
    n_haps_ref: usize,
}

impl HmmParams {
    pub fn new(variants: &[Variant], n_haps_ref: usize) -> Self {
        let eprob = (EPROB, 1. - EPROB);

        #[cfg(feature = "leak-resist")]
        let eprob = (Real::protect_f32(eprob), Real::protect_f32(rev_eprob));

        let forward_rprobs = compute_all_recomb_probs(&variants, N_EFF, n_haps_ref);

        Self {
            eprob: (eprob.0.into(), eprob.1.into()),
            forward_rprobs,
            n_haps_ref,
        }
    }

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> HmmParamsSlice<'a> {
        HmmParamsSlice {
            eprob: self.eprob,
            forward_rprobs: &self.forward_rprobs[start..end],
            backward_prob_last: self.forward_rprobs[end % self.forward_rprobs.len()],
            n_haps_ref: self.n_haps_ref,
        }
    }
}

pub struct HmmParamsSlice<'a> {
    pub eprob: (Real, Real),
    forward_rprobs: &'a [(Real, Real)],
    backward_prob_last: (Real, Real),
    n_haps_ref: usize,
}

impl<'a> HmmParamsSlice<'a> {
    pub fn get_forward_rprobs(&self, i: usize) -> (Real, Real) {
        self.forward_rprobs[i]
    }

    pub fn get_backward_rprobs(&self, i: usize) -> (Real, Real) {
        if i == self.forward_rprobs.len() - 1 {
            self.backward_prob_last
        } else {
            self.forward_rprobs[i + 1]
        }
    }

    pub fn get_rprobs_from_cm_dist(&self, dist_cm: Real) -> (Real, Real) {
        let r = compute_recomb_prob(dist_cm, N_EFF, self.n_haps_ref);
        (r, 1. - r)
    }
}

fn compute_all_recomb_probs(variants: &[Variant], n_eff: usize, n_haps: usize) -> Vec<(f64, f64)> {
    let mut recomb_probs = Vec::with_capacity(variants.len());
    recomb_probs.push((0., 1.));
    for (prev, cur) in variants.iter().zip(variants.iter().skip(1)) {
        let r = compute_recomb_prob((cur.cm - prev.cm) as f64, n_eff, n_haps);
        recomb_probs.push((r, 1. - r));
    }
    recomb_probs
}

#[inline]
fn compute_recomb_prob(dist_cm: f64, n_eff: usize, n_haps: usize) -> f64 {
    -1. * (-0.04 * n_eff as f64 * dist_cm.max(1e-7) / n_haps as f64).exp_m1()
}
