use crate::genotype_graph::{G, P};
use crate::{tp_convert_to, tp_value_real, RealHmm, UInt, U8};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView3};
use rand::Rng;

#[cfg(feature = "leak-resist-new")]
use timing_shield::{TpBool, TpEq, TpOrd, TpU32};

pub fn forward_sampling(
    prev_ind: (u8, u8),
    tprobs_dips: ArrayView3<RealHmm>,
    genotype_graph: ArrayView1<G>,
    is_first_window: bool,
    mut rng: impl Rng,
) -> Array2<U8> {
    let m = tprobs_dips.shape()[0];
    let mut phase_ind = Array2::<U8>::zeros((m, 2));

    for i in 0..m {
        let (prev_ind1, prev_ind2) = if i == 0 {
            (prev_ind.0, prev_ind.1)
        } else {
            (phase_ind[[i - 1, 0]], phase_ind[[i - 1, 1]])
        };

        let (ind1, ind2) = if genotype_graph[i].is_segment_marker() || (is_first_window && i == 0) {
            //let (ind1, ind2) = if genotype_graph[i].is_segment_marker() {
            constrained_paired_sample(
                tprobs_dips.slice(s![i, prev_ind1 as usize, ..]),
                tprobs_dips.slice(s![i, prev_ind2 as usize, ..]),
                &mut rng,
            )
        } else {
            (prev_ind1 as u32, prev_ind2 as u32)
        };
        phase_ind[[i, 0]] = tp_convert_to!(ind1, u8);
        phase_ind[[i, 1]] = tp_convert_to!(ind2, u8);
    }
    phase_ind
}

// Only pairs of indices that add up to n (length of the weight vectors) are allowed
// Weight of a pair is the product of the two weights (joint probability)
fn constrained_paired_sample(
    weights1: ArrayView1<RealHmm>,
    weights2: ArrayView1<RealHmm>,
    rng: impl Rng,
) -> (UInt, UInt) {
    let scale_1 = tp_value_real!(1, i64) / weights1.sum();
    let weights1 = &weights1 * scale_1;
    let scale_2 = tp_value_real!(1, i64) / weights2.sum();
    let weights2 = &weights2 * scale_2;
    let mut combined = Array1::<RealHmm>::zeros(P);
    for i in 0..P {
        combined[i] = weights1[i] * weights2[P - 1 - i];
    }
    let ind1 = weighted_sample(combined.view(), rng);
    //if ind1 >= 8 {
    //println!("{:?}", combined);
    //}
    assert!(ind1 < 8);
    (ind1, P as u32 - 1 - ind1)
}

fn weighted_sample(weights: ArrayView1<RealHmm>, mut rng: impl Rng) -> UInt {
    let mut total_weight = weights[0];
    let mut cumulative_weights: Vec<RealHmm> = Vec::with_capacity(weights.len());
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
