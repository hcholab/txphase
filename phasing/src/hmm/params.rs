use crate::Real;

const EPROB: f64 = 0.0001;
const N_EFF: f64 = 15000.;

pub struct HmmParams {
    pub eprob: Real,
    n_haps_ref_frac: Real,
}

impl HmmParams {
    pub fn new(n_haps_ref: usize) -> Self {
        Self {
            eprob: EPROB / (1. - EPROB),
            n_haps_ref_frac: 1. / n_haps_ref as f64,
        }
    }

    pub fn get_rprobs(&self, dist_cm: Real) -> (Real, Real) {
        let r = compute_recomb_prob(dist_cm, self.n_haps_ref_frac);
        (r * self.n_haps_ref_frac, (1. - r))
    }
}

#[inline]
fn compute_recomb_prob(dist_cm: f64, n_haps_frac: f64) -> f64 {
    -(-0.04 * N_EFF * dist_cm.max(1e-7) * n_haps_frac).exp_m1()
}
