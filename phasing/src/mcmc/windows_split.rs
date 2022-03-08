const MIN_N_VARIANTS_PER_SEGMENT: usize = 100;

use crate::genotype_graph::GenotypeGraph;
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
        let window_midpoint_cm = window_cm * 3. / 8. + rng.gen_range(0.0..window_cm / 4.);
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

pub fn split_by_segment(
    genotype_graph: &GenotypeGraph,
    variants: ArrayView1<Variant>,
    min_window_len_cm: f64,
    mut rng: impl Rng,
) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let mut start_segment_i = 0;
    for (i, g) in genotype_graph.graph.iter().enumerate() {
        if g.is_segment_marker() {
            segments.push(start_segment_i);
            start_segment_i = i;
        }
    }
    segments.push(start_segment_i);
    let mut boundaries = Vec::with_capacity(variants.len());
    let boundaries_starts =
        recursive_split_by_segment(&segments, variants, 0, min_window_len_cm, &mut rng).unwrap();
    for (&start, &end) in boundaries_starts
        .iter()
        .zip(boundaries_starts.iter().skip(1))
    {
        boundaries.push((start, end))
    }
    boundaries.push((*boundaries_starts.last().unwrap(), variants.len()));
    boundaries
}

fn recursive_split_by_segment(
    segments: &[usize],
    variants: ArrayView1<Variant>,
    base_index: usize,
    min_window_len_cm: f64,
    rng: &mut impl Rng,
) -> Option<Vec<usize>> {
    let window_cm = variants.last().unwrap().cm - variants.first().unwrap().cm;
    if variants.len() < MIN_N_VARIANTS_PER_SEGMENT
        || window_cm < min_window_len_cm
        || segments.len() < 4
    {
        None
    } else {
        let segment_mid_point = segments.len() / 4 + rng.gen_range(0..segments.len() / 2) + 1;
        let mid_point = segments[segment_mid_point] - base_index;
        let head = recursive_split_by_segment(
            &segments[..segment_mid_point],
            variants.slice(s![..mid_point]),
            base_index,
            min_window_len_cm,
            rng,
        );
        let tail = recursive_split_by_segment(
            &segments[segment_mid_point..],
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
