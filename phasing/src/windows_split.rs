const MIN_N_VARIANTS_PER_SEGMENT: usize = 100;

use crate::variants::Variant;

pub fn split(variants: &[Variant], min_window_len_cm: f64) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::with_capacity(variants.len());
    let boundaries_starts = recursive_split(variants, 0, min_window_len_cm);
    for (&start, &end) in boundaries_starts
        .iter()
        .zip(boundaries_starts.iter().skip(1))
    {
        boundaries.push((start, end))
    }
    boundaries.push((*boundaries_starts.last().unwrap(), variants.len()));

    boundaries
}

fn recursive_split(variants: &[Variant], base_index: usize, min_window_len_cm: f64) -> Vec<usize> {
    let window_cm = variants.last().unwrap().cm - variants.first().unwrap().cm;
    if variants.len() < MIN_N_VARIANTS_PER_SEGMENT || window_cm < min_window_len_cm {
        vec![base_index]
    } else {
        let mid_point = match variants.binary_search_by(|probe| {
            probe
                .cm
                .partial_cmp(&(variants.first().unwrap().cm + window_cm / 2.))
                .unwrap()
        }) {
            Ok(v) => v,
            Err(v) => v,
        };
        let mut head = recursive_split(&variants[..mid_point], base_index, min_window_len_cm);
        let tail = recursive_split(
            &variants[mid_point..],
            base_index + mid_point,
            min_window_len_cm,
        );
        head.extend_from_slice(&tail);
        head
    }
}
