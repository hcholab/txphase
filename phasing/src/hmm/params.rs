use crate::{RealHmm, tp_value_real, tp_expose_real};

const EPROB: f64 = 0.0001;
const N_EFF: f64 = 15000.;

pub struct HmmParams {
    pub eprob: RealHmm,
    n_haps_ref_frac: RealHmm,
}

impl HmmParams {
    pub fn new(n_haps_ref: usize) -> Self {
        Self {
            eprob: tp_value_real!(EPROB / (1. - EPROB), f32),
            n_haps_ref_frac: tp_value_real!(1. / n_haps_ref as f64, f32),
        }
    }

    pub fn get_rprobs(&self, dist_cm: RealHmm) -> (RealHmm, RealHmm) {
        let r = compute_recomb_prob(dist_cm, self.n_haps_ref_frac);
        (r * self.n_haps_ref_frac, (tp_value_real!(1, i64) - r))
    }
}

//TODO fix this
#[inline]
fn compute_recomb_prob(dist_cm: RealHmm, n_haps_frac: RealHmm) -> RealHmm {
    let dist_cm = tp_expose_real!(dist_cm) as f64;
    let n_haps_frac = tp_expose_real!(n_haps_frac) as f64;

    let r = -(-0.04 * N_EFF * dist_cm.max(1e-7) * n_haps_frac).exp_m1();

    tp_value_real!(r, f32)

}
