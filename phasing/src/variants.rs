pub struct Variant {
    pub cm: f64,
    afreq: f64,
    cref: usize,
    calt: usize,
}

impl Variant {
    pub fn new(cm: f64, afreq: f64, n_haps: usize) -> Self {
        let calt = (n_haps as f64 * afreq).round() as usize;
        let cref = n_haps - calt;
        Self {
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
}

pub fn build_variants(cms: &[f64], afreqs: &[f64], n_haps: usize) -> Vec<Variant> {
    afreqs
        .iter()
        .zip(cms.iter())
        .map(|(&afreq, &cm)| Variant::new(cm, afreq, n_haps))
        .collect()
}
