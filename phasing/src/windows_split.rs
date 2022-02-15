const MIN_N_VARIANTS_PER_SEGMENT: usize = 100;

use crate::variants::Variant;
use ndarray::{s, ArrayView1};
use rand::Rng;

pub fn split(
    variants: ArrayView1<Variant>,
    min_window_len_cm: f64,
    mut rng: impl Rng,
) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::with_capacity(variants.len());
    let boundaries_starts = recursive_split(variants, 0, min_window_len_cm, &mut rng).unwrap();
    for (&start, &end) in boundaries_starts
        .iter()
        .zip(boundaries_starts.iter().skip(1))
    {
        boundaries.push((start, end))
    }
    boundaries.push((*boundaries_starts.last().unwrap(), variants.len()));

    boundaries
}

fn recursive_split(
    variants: ArrayView1<Variant>,
    base_index: usize,
    min_window_len_cm: f64,
    rng: &mut impl Rng,
) -> Option<Vec<usize>> {
    let window_cm = variants.last().unwrap().cm - variants.first().unwrap().cm;
    if variants.len() < MIN_N_VARIANTS_PER_SEGMENT || window_cm < min_window_len_cm {
        None
    } else {
        let window_midpoint_cm = window_cm / 4. + rng.gen_range(0.0..window_cm / 2.);
        let mid_point = match variants.as_slice().unwrap().binary_search_by(|probe| {
            probe
                .cm
                .partial_cmp(&(variants.first().unwrap().cm + &window_midpoint_cm))
                .unwrap()
        }) {
            Ok(v) => v,
            Err(v) => v,
        };
        let head = recursive_split(
            variants.slice(s![..mid_point]),
            base_index,
            min_window_len_cm,
            rng,
        );
        let tail = recursive_split(
            variants.slice(s![mid_point..]),
            base_index + mid_point,
            min_window_len_cm,
            rng,
        );
        if head.is_none() || tail.is_none() {
            Some(vec![base_index])
        } else {
            let mut head = head.unwrap();
            head.extend_from_slice(&tail.unwrap());
            Some(head)
        }
    }
}
