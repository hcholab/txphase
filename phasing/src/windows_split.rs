const MIN_N_VARIANTS_PER_SEGMENT: usize = 100;

pub struct Windows {
    pub boundaries: Vec<(usize, usize)>,
    pub n_pos_window_overlap: usize,
}

impl Windows {
    pub fn new(mut cms: &[f32], min_window_len_cm: f32, n_pos_window_overlap: usize) -> Self {
        let mut boundaries = Vec::new();
        let mut start = 0;
        let mut end;
        loop {
            let idx = match cms[MIN_N_VARIANTS_PER_SEGMENT..]
                .binary_search_by(|probe| probe.partial_cmp(&(cms[0] + min_window_len_cm)).unwrap())
            {
                Ok(v) => v + 1,
                Err(v) => v + 1,
            };
            if MIN_N_VARIANTS_PER_SEGMENT + idx <= cms.len() {
                end = start + MIN_N_VARIANTS_PER_SEGMENT + idx;
                boundaries.push((start, end));
                cms = &cms[MIN_N_VARIANTS_PER_SEGMENT + idx - n_pos_window_overlap..];
                start = end - n_pos_window_overlap;
            } else {
                boundaries.push((start, start + cms.len()));
                break;
            }
        }

        Self {
            boundaries,
            n_pos_window_overlap,
        }
    }
}
