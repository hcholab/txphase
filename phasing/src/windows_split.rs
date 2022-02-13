const MIN_N_VARIANTS_PER_SEGMENT: usize = 100;

pub fn split(cms: &[f32], min_window_len_cm: f32) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::with_capacity(cms.len());
    let boundaries_starts = recursive_split(cms, 0, min_window_len_cm);
    for (&start, &end) in boundaries_starts
        .iter()
        .zip(boundaries_starts.iter().skip(1))
    {
        boundaries.push((start, end))
    }
    boundaries.push((*boundaries_starts.last().unwrap(), cms.len()));

    boundaries
}

fn recursive_split(cms: &[f32], base_index: usize, min_window_len_cm: f32) -> Vec<usize> {
    let window_cm = cms.last().unwrap() - cms.first().unwrap();
    if cms.len() < MIN_N_VARIANTS_PER_SEGMENT || window_cm < min_window_len_cm {
        vec![base_index]
    } else {
        let mid_point = match cms.binary_search_by(|probe| {
            probe
                .partial_cmp(&(cms.first().unwrap() + window_cm / 2.))
                .unwrap()
        }) {
            Ok(v) => v,
            Err(v) => v,
        };
        let mut head = recursive_split(&cms[..mid_point], base_index, min_window_len_cm);
        let tail = recursive_split(&cms[mid_point..], base_index + mid_point, min_window_len_cm);
        head.extend_from_slice(&tail);
        head
    }
}
