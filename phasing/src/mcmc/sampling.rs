use crate::genotype_graph::{G, P};
use crate::{tp_value_real, Real, U8};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView3};
use rand::Rng;

#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpI16, TpOrd};

pub fn forward_sampling(
    prev_ind: (U8, U8),
    tprobs: ArrayView3<Real>,
    #[cfg(feature = "obliv")] tprobs_e: ArrayView3<TpI16>,
    genotype_graph: ArrayView1<G>,
    is_first_window: bool,
    #[cfg(feature = "obliv")] start_i: crate::Usize,
    mut rng: impl Rng,
) -> Array2<U8> {
    let m = tprobs.shape()[0];

    #[cfg(feature = "obliv")]
    let mut phase_ind = Array2::<U8>::from_elem((m, 2), U8::protect(0));

    #[cfg(not(feature = "obliv"))]
    let mut phase_ind = Array2::<U8>::zeros((m, 2));

    for i in 0..m {
        #[cfg(feature = "obliv")]
        let (prev_ind1, prev_ind2) = {
            let cond = (i == 0) | start_i.tp_eq(&(i as u64));
            let i = if i == 0 { 1 } else { i };
            (
                cond.select(prev_ind.0, phase_ind[[i - 1, 0]]),
                cond.select(prev_ind.1, phase_ind[[i - 1, 1]]),
            )
        };

        #[cfg(not(feature = "obliv"))]
        let (prev_ind1, prev_ind2) = if i == 0 {
            (prev_ind.0, prev_ind.1)
        } else {
            (phase_ind[[i - 1, 0]], phase_ind[[i - 1, 1]])
        };

        #[cfg(feature = "obliv")]
        let (ind1, ind2) = {
            let mut weights1 = [Real::ZERO; P];
            let mut weights1_e = [TpI16::protect(0); P];
            let mut weights2 = [Real::ZERO; P];
            let mut weights2_e = [TpI16::protect(0); P];

            for j in 0..P {
                let cond1 = prev_ind1.tp_eq(&(j as u8));
                weights1
                    .iter_mut()
                    .zip(&tprobs.slice(s![i, j, ..]))
                    .for_each(|(w, &p)| *w = cond1.select(p, *w));
                weights1_e
                    .iter_mut()
                    .zip(&tprobs_e.slice(s![i, j, ..]))
                    .for_each(|(w, &p)| *w = cond1.select(p, *w));
                let cond2 = prev_ind2.tp_eq(&(j as u8));
                weights2
                    .iter_mut()
                    .zip(&tprobs.slice(s![i, j, ..]))
                    .for_each(|(w, &p)| *w = cond2.select(p, *w));
                weights2_e
                    .iter_mut()
                    .zip(&tprobs_e.slice(s![i, j, ..]))
                    .for_each(|(w, &p)| *w = cond2.select(p, *w));
            }
            let (ind1, ind2) = constrained_paired_sample(
                ArrayView1::from(&weights1),
                ArrayView1::from(&weights1_e),
                ArrayView1::from(&weights2),
                ArrayView1::from(&weights2_e),
                &mut rng,
            );

            let cond = genotype_graph[i].is_segment_marker()
                | (is_first_window & start_i.tp_eq(&(i as u64)));
            (cond.select(ind1, prev_ind1), cond.select(ind2, prev_ind2))
        };
        #[cfg(not(feature = "obliv"))]
        let (ind1, ind2) = if genotype_graph[i].is_segment_marker() || (is_first_window && i == 0) {
            constrained_paired_sample(
                tprobs.slice(s![i, prev_ind1 as usize, ..]),
                tprobs.slice(s![i, prev_ind2 as usize, ..]),
                &mut rng,
            )
        } else {
            (prev_ind1, prev_ind2)
        };

        phase_ind[[i, 0]] = ind1;
        phase_ind[[i, 1]] = ind2;
    }
    phase_ind
}

// Only pairs of indices that add up to n (length of the weight vectors) are allowed
// Weight of a pair is the product of the two weights (joint probability)
fn constrained_paired_sample(
    weights1: ArrayView1<Real>,
    #[cfg(feature = "obliv")] weights1_e: ArrayView1<TpI16>,
    weights2: ArrayView1<Real>,
    #[cfg(feature = "obliv")] weights2_e: ArrayView1<TpI16>,
    rng: impl Rng,
) -> (U8, U8) {
    #[cfg(not(feature = "obliv"))]
    let (weights1, weights2) = {
        let weights1 = &weights1 / weights1.sum();
        let weights2 = &weights2 / weights2.sum();
        (weights1, weights2)
    };

    let mut combined = Array1::<Real>::zeros(P);
    #[cfg(feature = "obliv")]
    let mut combined_e = Array1::<TpI16>::from_elem(P, TpI16::protect(0));
    for i in 0..P {
        combined[i] = weights1[i] * weights2[P - 1 - i];
        #[cfg(feature = "obliv")]
        {
            combined_e[i] = weights1_e[i] + weights2_e[P - 1 - i];
        }
    }

    #[cfg(feature = "obliv")]
    crate::hmm::renorm_equalize_scale_arr1(combined.view_mut(), combined_e.view_mut());

    let ind1 = weighted_sample(combined.view(), rng);

    #[cfg(feature = "obliv")]
    return (ind1, P as u8 - 1 - ind1);

    #[cfg(not(feature = "obliv"))]
    (ind1 as u8, P as u8 - 1 - ind1 as u8)
}

fn weighted_sample(weights: ArrayView1<Real>, mut rng: impl Rng) -> U8 {
    let mut total_weight = weights[0];
    let mut cumulative_weights: Vec<Real> = Vec::with_capacity(weights.len());
    cumulative_weights.push(total_weight);
    for &w in weights.iter().skip(1) {
        total_weight += w;
        cumulative_weights.push(total_weight);
    }

    #[cfg(feature = "obliv")]
    let chosen_weight = tp_value_real!(rng.gen_range(0.0..1.0), f32) * total_weight;

    #[cfg(not(feature = "obliv"))]
    let chosen_weight = tp_value_real!(rng.gen_range(0.0..total_weight), f32);

    #[cfg(feature = "obliv")]
    {
        let mut index = U8::protect(0);
        let mut done = TpBool::protect(false);
        for w in cumulative_weights {
            done = w
                .tp_gt_eq(&chosen_weight)
                .select(TpBool::protect(true), done);
            index = done.select(index, index + 1);
        }
        index
            .tp_eq(&(weights.len() as u8))
            .select(U8::protect(0), index)
    }

    #[cfg(not(feature = "obliv"))]
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
            .unwrap_err() as u8
    }
}
