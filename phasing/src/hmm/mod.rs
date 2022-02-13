mod params;
pub use params::*;

use crate::genotypes::{Genotypes, G, P};
use crate::{tp_value, Genotype, Real};

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, Zip};

fn backward(
    ref_panel: ArrayView2<Genotype>,
    target: &Genotypes,
    hmm_params: &HmmParamsSlice,
) -> Array3<Real> {
    let m = ref_panel.nrows();
    let n = ref_panel.ncols();

    let mut bprobs = unsafe { Array3::<Real>::uninit((m, P, n)).assume_init() };

    // initialization (uniform over ref haplotypes + emission at last position)
    emission_init(
        ref_panel.row(m - 1),
        target.genotypes[m - 1],
        hmm_params,
        bprobs.slice_mut(s![m - 1, .., ..]),
    );

    renormalize(bprobs.slice_mut(s![m - 1, .., ..]));

    let uniform_frac = 1.0 / (n as f32);

    #[cfg(feature = "leak-resist")]
    let uniform_frac = Real::protect_f32(uniform_frac);

    for i in (0..m - 1).rev() {
        //  transition from i+1 -> i
        let (cur_bprob, prev_bprob) = bprobs.multi_slice_mut((s![i, .., ..], s![i + 1, .., ..]));
        let prev_target = target.genotypes[i + 1];
        transition(
            prev_target,
            hmm_params.rprobs[i],
            uniform_frac.into(),
            cur_bprob,
            prev_bprob.view(),
        );

        // emission at i
        emission(
            ref_panel.row(i),
            target.genotypes[i],
            hmm_params,
            bprobs.slice_mut(s![i, .., ..]),
        );
        renormalize(bprobs.slice_mut(s![i, .., ..]));
    }

    bprobs
}

pub fn forward_backward(
    ref_panel: ArrayView2<Genotype>,
    target: &Genotypes,
    hmm_params: &HmmParamsSlice,
) -> Array3<Real> {
    let m = ref_panel.nrows();
    let n = ref_panel.ncols();

    let bprobs = backward(ref_panel, target, hmm_params);

    // p x p matrix at each pos i for transition between i-1 and i
    // Save belief over the first block in tprob[0][0]
    let firstprob = Array1::<Real>::from_shape_fn(P, |i| bprobs.slice(s![0, i, ..]).iter().sum());

    let mut tprob = unsafe { Array3::<Real>::uninit((m, P, P)).assume_init() };

    for i in 0..P {
        tprob.slice_mut(s![0, i, ..]).assign(&firstprob);
    }

    // Initialize forward prob
    let mut prev_fprobs = unsafe { Array2::<Real>::uninit((P, n)).assume_init() };
    let mut cur_fprobs = unsafe { Array2::<Real>::uninit((P, n)).assume_init() };

    // Emission at first position
    emission_init(
        ref_panel.row(0),
        target.genotypes[0],
        hmm_params,
        prev_fprobs.view_mut(),
    );

    let uniform_frac = 1.0 / (n as f32);

    #[cfg(feature = "leak-resist")]
    let uniform_frac = Real::protect_f32(uniform_frac);

    for i in 1..m {
        let prev_target = target.genotypes[i - 1];
        transition(
            prev_target,
            hmm_params.rprobs[i],
            uniform_frac.into(),
            cur_fprobs.view_mut(),
            prev_fprobs.view(),
        );

        // Combine fprob (from i-1) and bprob[i] to get transition probs
        let mut weights = unsafe { Array2::<Real>::uninit((P, P)).assume_init() };
        for j in 0..P {
            // Multiply with backward prob and integrate to get transition prob from hap j to h1
            for h1 in 0..P {
                // If not between blocks, set to identity matrix
                let ind = tp_value!(j, u8);

                //#[cfg(feature = "leak-resist")]
                //{
                //let iprod = inner_prod(cur_fprobs.row(j), bprobs.slice(s![i, h1, ..]));
                //#[cfg(feature = "leak-resist-fast")]
                //{
                //weights[[j, h1]] = ind.tp_eq(&U8::protect(h1 as u8)).select(
                //self.block_head[i].select(iprod, Real::protect_f32(1.0)),
                //self.block_head[i].select(iprod, Real::protect_f32(0.0)),
                //);
                //}
                //#[cfg(not(feature = "leak-resist-fast"))]
                //{
                //weights[[j, h1]] = ind.tp_eq(&U8::protect(h1 as u8)).select(
                //self.block_head[i].select(iprod, Real::protect_f32(1.0)),
                //self.block_head[i].select(iprod, Real::NAN),
                //);
                //}
                //}

                //#[cfg(not(feature = "leak-resist"))]
                //{
                weights[[j, h1]] = if target.genotypes[i].is_segment_marker() {
                    inner_prod(cur_fprobs.row(ind.into()), bprobs.slice(s![i, h1, ..]))
                } else {
                    if j as usize == h1 {
                        1.0
                    } else {
                        0.0
                    }
                };
                //}
            }
        }

        tprob.slice_mut(s![i, .., ..]).assign(&weights);

        // Add emission at i (with the sampled haplotypes) to fprob_next
        emission(
            ref_panel.row(i),
            target.genotypes[i],
            hmm_params,
            cur_fprobs.view_mut(),
        );

        // renormalize
        renormalize(cur_fprobs.view_mut());

        // Update fprob
        prev_fprobs.assign(&cur_fprobs);
    }

    tprob
}

fn emission(
    ref_panel: ArrayView1<Genotype>,
    target: G,
    hmm_params: &HmmParamsSlice,
    mut probs: ArrayViewMut2<Real>,
) {
    let mut eprobs = unsafe { Array2::<Real>::uninit(probs.dim()).assume_init() };
    emission_init(ref_panel, target, hmm_params, eprobs.view_mut());
    probs *= &eprobs;
}

fn emission_init(
    ref_panel: ArrayView1<Genotype>,
    target: G,
    hmm_params: &HmmParamsSlice,
    mut probs_at_pos: ArrayViewMut2<Real>,
) {
    Zip::indexed(probs_at_pos.rows_mut()).for_each(|i, mut a| {
        let b = target.access_graph_row(i as u8);
        Zip::from(&mut a).and(ref_panel).for_each(|c, &d| {
            *c = emission_prob(d, b, hmm_params.eprob);
        });
    });
}

#[inline(always)]
fn emission_prob(
    x_geno: Genotype,
    t_geno: Genotype,
    error_prob: (Real, Real), // (e, 1-e)
) -> Real {
    #[cfg(feature = "leak-resist")]
    {
        x_geno.tp_eq(&t_geno).select(error_prob.1, error_prob.0)
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        if x_geno == t_geno {
            error_prob.1
        } else {
            error_prob.0
        }
    }
}

// transition from previos to current position
fn transition(
    prev_target: G,
    rprob: (Real, Real),
    uniform_frac: Real,
    mut cur_probs: ArrayViewMut2<Real>,
    prev_probs: ArrayView2<Real>,
) {
    Zip::from(cur_probs.rows_mut())
        .and(prev_probs.rows())
        .for_each(|mut a, b| {
            let sum: Real = b.iter().sum();
            Zip::from(&mut a).and(&b).for_each(|c, &d| {
                *c = transition_prob(d, sum, uniform_frac.into(), rprob);
            });
        });

    // segment transition
    // Add to aggregate sum
    let sums = Array1::from_shape_fn(cur_probs.ncols(), |i| cur_probs.column(i).iter().sum());
    Zip::from(cur_probs.rows_mut()).for_each(|mut a| {
        Zip::from(&mut a).and(&sums).for_each(|b, c| {
            //#[cfg(feature = "leak-resist")]
            //{
            //*b = next_block_head.select(*c, *b);
            //}
            //#[cfg(not(feature = "leak-resist"))]
            //{
            if prev_target.is_segment_marker() {
                *b = *c;
            }
            //}
        });
    });
}

#[inline(always)]
fn transition_prob(
    prev_prob: Real,
    total_prob: Real,
    uniform_frac: Real,
    recomb_prob: (Real, Real), // (r, 1-r)
) -> Real {
    prev_prob * recomb_prob.1 + total_prob * recomb_prob.0 * uniform_frac
}

fn renormalize(mut probs_at_pos: ArrayViewMut2<Real>) {
    let sum: Real = probs_at_pos.iter().sum();
    probs_at_pos /= sum;
}

fn inner_prod(v1: ArrayView1<Real>, v2: ArrayView1<Real>) -> Real {
    let prod = &v1 * &v2;
    prod.into_iter().sum()
}
