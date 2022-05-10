use crate::genotype_graph::{G, P};
use crate::{tp_convert_to, tp_value_real, Real, UInt, U8};
use ndarray::{s, Array1, ArrayView1, ArrayView3};
use rand::Rng;

#[cfg(feature = "leak-resist-new")]
use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpI16, TpOrd, TpU32};

pub fn forward_sampling(
    prev_ind: u8,
    tprobs: ArrayView3<Real>,
    #[cfg(feature = "leak-resist-new")] tprobs_e: ArrayView3<TpI16>,
    genotype_graph: ArrayView1<G>,
    is_first_window: bool,
    mut rng: impl Rng,
) -> Array1<U8> {
    let m = tprobs.shape()[0];
    let mut phase_ind = Array1::<U8>::zeros(m);

    for i in 0..m {
        let prev_ind = if i == 0 {
            prev_ind
        } else {
            phase_ind[i - 1]
        };

        let cur_ind = if genotype_graph[i].is_segment_marker() || (is_first_window && i == 0) {
            let i = constrained_paired_sample(
                tprobs.slice(s![i, prev_ind as usize, ..]),
                #[cfg(feature = "leak-resist-new")]
                tprobs_e.slice(s![i, prev_ind as usize, ..]),
                tprobs.slice(s![i, P - 1 - prev_ind as usize, ..]),
                #[cfg(feature = "leak-resist-new")]
                tprobs_e.slice(s![i, P - 1 - prev_ind as usize, ..]),
                &mut rng,
            );
            i
        } else {
            prev_ind as u32
        };
        phase_ind[i] = tp_convert_to!(cur_ind, u8);
    }
    phase_ind
}

// Only pairs of indices that add up to n (length of the weight vectors) are allowed
// Weight of a pair is the product of the two weights (joint probability)
pub fn constrained_paired_sample(
    weights1: ArrayView1<Real>,
    #[cfg(feature = "leak-resist-new")] weights1_e: ArrayView1<TpI16>,
    weights2: ArrayView1<Real>,
    #[cfg(feature = "leak-resist-new")] weights2_e: ArrayView1<TpI16>,
    rng: impl Rng,
) -> UInt {
    #[cfg(not(feature = "leak-resist-new"))]
    let (weights1, weights2) = (
        &weights1 * 1. / weights1.sum(),
        &weights2 * 1. / weights2.sum(),
    );

    let mut combined = Array1::<Real>::from_shape_fn(P, |i| weights1[i] * weights2[P - 1 - i]);

    #[cfg(feature = "leak-resist-new")]
    let mut combined_e = Array1::<TpI16>::from_shape_fn(P, |i| weights1_e[i] + weights2_e[P - 1 - i]);

    #[cfg(feature = "leak-resist-new")]
    crate::hmm::renorm_equalize_scale_arr1(combined.view_mut(), combined_e.view_mut());

    let ind = weighted_sample(combined.view(), rng);

    debug_assert!(ind < 8);
    ind
}

fn weighted_sample(weights: ArrayView1<Real>, mut rng: impl Rng) -> UInt {
    let mut total_weight = weights[0];
    let mut cumulative_weights: Vec<Real> = Vec::with_capacity(weights.len());
    cumulative_weights.push(total_weight);
    for &w in weights.iter().skip(1) {
        total_weight += w;
        cumulative_weights.push(total_weight);
    }

    let chosen_weight = tp_value_real!(rng.gen_range(0.0..1.0), f32) * total_weight;

    #[cfg(feature = "leak-resist-new")]
    {
        let mut index = TpU32::protect(0);
        let mut done = TpBool::protect(false);
        for w in cumulative_weights {
            done = w.tp_gt(&chosen_weight).select(TpBool::protect(true), done);
            index = (!done).select(index + 1, index);
        }
        index
            .tp_eq(&(weights.len() as u32))
            .select(TpU32::protect(0), index)
            .expose()
    }

    #[cfg(not(feature = "leak-resist-new"))]
    {
        use std::cmp::Ordering;
        cumulative_weights
            .binary_search_by(|w| {
                if *w <= chosen_weight {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_err() as u32
    }
}
