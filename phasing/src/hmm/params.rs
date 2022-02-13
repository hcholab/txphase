use crate::Real;

const EPROB: f32 = 0.0001;
const N_EFF: usize = 15000;

pub struct HmmParams {
    pub eprob: (Real, Real),       // (e, 1 - e)
    pub rprobs: Vec<(Real, Real)>, //(r, 1 - r)
}

impl HmmParams {
    pub fn new(cms: &[f32], n_haps_ref: usize) -> Self {
        let eprob = (EPROB, 1. - EPROB);

        #[cfg(feature = "leak-resist")]
        let eprob = (Real::protect_f32(eprob), Real::protect_f32(rev_eprob));

        let rprobs = compute_all_recomb_probs(&cms, N_EFF, n_haps_ref);

        Self {
            eprob: (eprob.0.into(), eprob.1.into()),
            rprobs,
        }
    }

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> HmmParamsSlice<'a> {
        HmmParamsSlice {
            eprob: self.eprob,
            rprobs: &self.rprobs[start..end],
        }
    }
}

pub struct HmmParamsSlice<'a> {
    pub eprob: (Real, Real),
    pub rprobs: &'a [(Real, Real)],
}

fn compute_all_recomb_probs(cms: &[f32], n_eff: usize, n_haps: usize) -> Vec<(f64, f64)> {
    let mut recomb_probs = Vec::with_capacity(cms.len());
    recomb_probs.push((0., 1.));
    for (prev, cur) in cms.iter().zip(cms.iter().skip(1)) {
        let r = compute_recomb_prob((cur - prev) as f64, n_eff, n_haps);
        recomb_probs.push((r, 1. - r));
    }
    recomb_probs
}

#[inline]
fn compute_recomb_prob(dist_cm: f64, n_eff: usize, n_haps: usize) -> f64 {
    -1. * (-0.04 * n_eff as f64 * dist_cm / n_haps as f64).exp_m1()
}
