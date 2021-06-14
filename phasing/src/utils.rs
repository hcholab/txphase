use rand::Rng;

pub fn gen_ref_panel(n_samples: usize, n_markers: usize, mut rng: impl Rng) -> Vec<Vec<u8>> {
    (0..n_markers)
        .into_iter()
        .map(|_| {
            (0..n_samples)
                .into_iter()
                .map(|_| rng.gen_range(0..2))
                .collect()
        })
        .collect()
}

pub fn gen_target_sample(n_markers: usize, mut rng: impl Rng) -> Vec<u8> {
    (0..n_markers)
        .into_iter()
        .map(|_| rng.gen_range(0..2))
        .collect()
}
 
