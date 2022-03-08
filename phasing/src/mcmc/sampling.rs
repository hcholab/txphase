use crate::genotype_graph::G;
use crate::{tp_convert_to, tp_value, Real, UInt, U8};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView3};
use rand::Rng;

pub fn forward_sampling(
    prev_ind: (u8, u8),
    tprobs: ArrayView3<Real>,
    genotype_graph: ArrayView1<G>,
    mut rng: impl Rng,
) -> Array2<U8> {
    let m = tprobs.shape()[0];
    let mut phase_ind = unsafe { Array2::<U8>::uninit((m, 2)).assume_init() };

    for i in 0..m {
        let (prev_ind1, prev_ind2) = if i == 0 {
            (prev_ind.0, prev_ind.1)
        } else {
            (phase_ind[[i - 1, 0]], phase_ind[[i - 1, 1]])
        };


        let (ind1, ind2) =  if genotype_graph[i].is_segment_marker() {
            constrained_paired_sample(
                tprobs.slice(s![i, prev_ind1 as usize, ..]),
                tprobs.slice(s![i, prev_ind2 as usize, ..]),
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
    weights1: ArrayView1<Real>,
    weights2: ArrayView1<Real>,
    rng: impl Rng,
) -> (UInt, UInt) {
    let weights1 = &weights1 / weights1.sum();
    let weights2 = &weights2 / weights2.sum();
    let n = weights1.len();
    let mut combined = unsafe { Array1::<Real>::uninit(n).assume_init() };
    for i in 0..n {
        #[cfg(feature = "leak-resist")]
        {
            #[cfg(not(feature = "leak-resist-fast"))]
            {
                combined[i] = (weights1[i].is_nan() | weights2[n - 1 - i].is_nan())
                    .select(Real::NAN, weights1[i] * weights2[n - 1 - i]);
            }
            #[cfg(feature = "leak-resist-fast")]
            {
                combined[i] = weights1[i] * weights2[n - 1 - i];
            }
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            combined[i] = weights1[i] * weights2[n - 1 - i];
        }
    }
    let ind1 = weighted_sample(combined.view(), rng);
    assert!(ind1 < 8);
    (ind1, tp_value!(n, u32) - 1 - ind1)
}

fn weighted_sample(weights: ArrayView1<Real>, mut rng: impl Rng) -> UInt {
    let mut total_weight = weights[0];
    let mut cumulative_weights: Vec<Real> = Vec::with_capacity(weights.len());
    cumulative_weights.push(total_weight);
    for &w in weights.iter().skip(1) {
        #[cfg(feature = "leak-resist")]
        {
            #[cfg(not(feature = "leak-resist-fast"))]
            {
                total_weight = w.is_nan().select(
                    total_weight,
                    total_weight.is_nan().select(w, total_weight + w),
                );
            }
            #[cfg(feature = "leak-resist-fast")]
            {
                total_weight += w;
            }
        }
        #[cfg(not(feature = "leak-resist"))]
        {
            total_weight += w;
        }
        cumulative_weights.push(total_weight);
    }

    #[cfg(feature = "leak-resist")]
    {
        let chosen_weight = Real::protect_f32(rng.gen_range(0.0..1.0)) * total_weight;
        let mut index = UInt::protect(0);
        let mut done = Bool::protect(false);
        for w in cumulative_weights {
            #[cfg(not(feature = "leak-resist-fast"))]
            {
                done = (w.tp_gt(&chosen_weight) & !w.is_nan()).select(Bool::protect(true), done);
            }
            #[cfg(feature = "leak-resist-fast")]
            {
                done = w.tp_gt(&chosen_weight).select(Bool::protect(true), done);
            }
            index = (!done).select(index + 1, index);
        }
        index
            .tp_eq(&(weights.len() as u32))
            .select(UInt::protect(0), index)
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        let chosen_weight = rng.gen_range(0.0..1.0) * total_weight;
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
