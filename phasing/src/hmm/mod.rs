mod params;
pub use params::*;

use crate::genotype_graph::{G, P};
use crate::{Genotype, Real};

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, Zip};

pub fn forward_backward(
    ref_panel: ArrayView2<Genotype>,
    genograph: ArrayView1<G>,
    hmm_params: &HmmParamsSlice,
) -> Array3<Real> {
    let m = ref_panel.nrows();
    let n = ref_panel.ncols();

    let bprobs = backward(ref_panel, genograph, hmm_params);

    let mut tprobs = unsafe { Array3::<Real>::uninit((m, P, P)).assume_init() };

    {
        // p x p matrix at each pos i for transition between i-1 and i
        // Save belief over the first block in tprob[0][0]
        let init_probs = Array1::<Real>::from_shape_fn(P, |i| bprobs.slice(s![0, i, ..]).sum());
        Zip::from(tprobs.slice_mut(s![0, .., ..]).rows_mut())
            .for_each(|mut r| r.assign(&init_probs));
    }

    // Initialize forward prob
    let mut prev_fprobs = unsafe { Array2::<Real>::uninit((P, n)).assume_init() };
    let mut cur_fprobs = unsafe { Array2::<Real>::uninit((P, n)).assume_init() };

    // Emission at first position
    emission_init(
        ref_panel.slice(s![0, ..]),
        genograph[0],
        hmm_params,
        prev_fprobs.view_mut(),
    );

    let uniform_frac = 1.0 / (n as f32);

    #[cfg(feature = "leak-resist")]
    let uniform_frac = Real::protect_f32(uniform_frac);

    for i in 1..m {
        transition(
            genograph[i - 1].is_segment_marker(),
            hmm_params.get_forward_rprobs(i),
            uniform_frac.into(),
            cur_fprobs.view_mut(),
            prev_fprobs.view(),
        );

        // Combine fprob (from i-1) and bprob[i] to get transition probs
        combine(
            cur_fprobs.view(),
            bprobs.slice(s![i, .., ..]),
            genograph[i].is_segment_marker(),
            tprobs.slice_mut(s![i, .., ..]),
        );

        // Add emission at i (with the sampled haplotypes) to fprob_next
        emission(
            ref_panel.slice(s![i, ..]),
            genograph[i],
            hmm_params,
            cur_fprobs.view_mut(),
        );

        // Update fprob
        prev_fprobs.assign(&cur_fprobs);
    }

    tprobs
}

fn backward(
    ref_panel: ArrayView2<Genotype>,
    genograph: ArrayView1<G>,
    hmm_params: &HmmParamsSlice,
) -> Array3<Real> {
    let m = ref_panel.nrows();
    let n = ref_panel.ncols();

    let mut bprobs = unsafe { Array3::<Real>::uninit((m, P, n)).assume_init() };

    // initialization (uniform over ref haplotypes + emission at last position)
    emission_init(
        ref_panel.row(m - 1),
        genograph[m - 1],
        hmm_params,
        bprobs.slice_mut(s![m - 1, .., ..]),
    );

    let uniform_frac = 1.0 / (n as f32);

    #[cfg(feature = "leak-resist")]
    let uniform_frac = Real::protect_f32(uniform_frac);

    for i in (0..m - 1).rev() {
        let (mut cur_bprob, prev_bprob) =
            bprobs.multi_slice_mut((s![i, .., ..], s![i + 1, .., ..]));

        transition(
            genograph[i + 1].is_segment_marker(),
            hmm_params.get_backward_rprobs(i),
            uniform_frac.into(),
            cur_bprob.view_mut(),
            prev_bprob.view(),
        );

        emission(
            ref_panel.row(i),
            genograph[i],
            hmm_params,
            bprobs.slice_mut(s![i, .., ..]),
        );
    }

    bprobs
}

fn emission(
    ref_panel: ArrayView1<Genotype>,
    graph_col: G,
    hmm_params: &HmmParamsSlice,
    mut probs: ArrayViewMut2<Real>,
) {
    Zip::indexed(probs.rows_mut()).for_each(|i, mut a| {
        Zip::from(&mut a).and(ref_panel).for_each(|c, &d| {
            *c *= emission_prob(d, graph_col.get_row(i), hmm_params.eprob);
        });
    });

    renormalize(probs);
}

fn emission_init(
    ref_panel: ArrayView1<Genotype>,
    graph_col: G,
    hmm_params: &HmmParamsSlice,
    mut probs: ArrayViewMut2<Real>,
) {
    Zip::indexed(probs.rows_mut()).for_each(|i, mut a| {
        Zip::from(&mut a).and(ref_panel).for_each(|c, &d| {
            *c = emission_prob(d, graph_col.get_row(i), hmm_params.eprob);
        });
    });

    renormalize(probs);
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

fn renormalize(mut probs: ArrayViewMut2<Real>) {
    let sum: Real = probs.sum();
    probs /= sum;
}

fn transition(
    prev_segment_marker: bool,
    rprob: (Real, Real),
    uniform_frac: Real,
    mut cur_probs: ArrayViewMut2<Real>,
    prev_probs: ArrayView2<Real>,
) {
    Zip::from(cur_probs.rows_mut())
        .and(prev_probs.rows())
        .for_each(|mut a, b| {
            let sum: Real = b.sum();
            Zip::from(&mut a).and(&b).for_each(|c, &d| {
                *c = transition_prob(d, sum, uniform_frac.into(), rprob);
            });
        });

    // segment transition
    // Add to aggregate sum
    let sums = Array1::from_shape_fn(cur_probs.ncols(), |i| cur_probs.column(i).sum());
    Zip::from(cur_probs.rows_mut()).for_each(|mut a| {
        Zip::from(&mut a).and(&sums).for_each(|b, c| {
            //#[cfg(feature = "leak-resist")]
            //{
            //*b = next_block_head.select(*c, *b);
            //}
            //#[cfg(not(feature = "leak-resist"))]
            //{
            if prev_segment_marker {
                *b = *c;
            }
            //}
        });
    });
}

#[inline]
fn transition_prob(
    prev_prob: Real,
    total_prob: Real,
    uniform_frac: Real,
    recomb_prob: (Real, Real), // (r, 1-r)
) -> Real {
    prev_prob * recomb_prob.1 + total_prob * recomb_prob.0 * uniform_frac
}

fn combine(
    fprobs: ArrayView2<Real>,
    bprobs: ArrayView2<Real>,
    is_segment_marker: bool,
    mut tprobs: ArrayViewMut2<Real>,
) {
    Zip::indexed(tprobs.rows_mut())
        .and(fprobs.rows())
        .for_each(|i, mut t_row, f_row| {
            Zip::indexed(&mut t_row)
                .and(bprobs.rows())
                .for_each(|j, t, b_row| {
                    *t = if is_segment_marker {
                        f_row.dot(&b_row)
                    } else {
                        if i == j {
                            1.
                        } else {
                            0.
                        }
                    };
                });
        })
}
