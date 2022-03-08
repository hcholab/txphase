mod params;
pub use params::*;

use crate::genotype_graph::{G, P};
use crate::variants::Variant;
use crate::{Genotype, Real};

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, Zip};

pub fn forward_backward(
    ref_panel: ArrayView2<Genotype>,
    genograph: ArrayView1<G>,
    hmm_params: &HmmParamsSlice,
    variants: ArrayView1<Variant>,
    ignored_sites: ArrayView1<bool>,
) -> Array3<Real> {
    let m = ref_panel.nrows();
    let n = ref_panel.ncols();

    let bprobs = backward(ref_panel, genograph, hmm_params, variants, ignored_sites);

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
    init(
        ref_panel.slice(s![0, ..]),
        genograph[0],
        hmm_params,
        prev_fprobs.view_mut(),
    );

    let uniform_frac = 1.0 / (n as f32);

    //#[cfg(feature = "leak-resist")]
    //let uniform_frac = Real::protect_f32(uniform_frac);

    let mut last_cm_dist = None;
    let mut imm_fprobs = unsafe { Array2::<Real>::uninit((P, n)).assume_init() };

    for i in 1..m {
        if !ignored_sites[i] {
            let rprobs = if let Some(last_cm_dist) = last_cm_dist {
                let cm_dist = variants[i].cm - last_cm_dist;
                hmm_params.get_rprobs_from_cm_dist(cm_dist)
            } else {
                hmm_params.get_forward_rprobs(i)
            };
            last_cm_dist = None;

            if genograph[i].is_segment_marker() {
                transition(
                    rprobs,
                    uniform_frac.into(),
                    imm_fprobs.view_mut(),
                    prev_fprobs.view(),
                );

                combine(
                    imm_fprobs.view(),
                    bprobs.slice(s![i, .., ..]),
                    tprobs.slice_mut(s![i, .., ..]),
                );

                collapse(
                    rprobs,
                    uniform_frac.into(),
                    cur_fprobs.view_mut(),
                    prev_fprobs.view(),
                );
            } else {
                transition(
                    rprobs,
                    uniform_frac.into(),
                    cur_fprobs.view_mut(),
                    prev_fprobs.view(),
                );
            }

            emission(
                ref_panel.slice(s![i, ..]),
                genograph[i],
                hmm_params,
                cur_fprobs.view_mut(),
            );

            prev_fprobs.assign(&cur_fprobs);
        } else {
            assert!(!genograph[i].is_segment_marker());
            if last_cm_dist.is_none() {
                last_cm_dist = Some(variants[i - 1].cm);
            }
        }
    }

    tprobs
}

fn backward(
    ref_panel: ArrayView2<Genotype>,
    genograph: ArrayView1<G>,
    hmm_params: &HmmParamsSlice,
    variants: ArrayView1<Variant>,
    ignored_sites: ArrayView1<bool>,
) -> Array3<Real> {
    let m = ref_panel.nrows();
    let n = ref_panel.ncols();

    let mut bprobs = unsafe { Array3::<Real>::uninit((m, P, n)).assume_init() };

    // initialization (uniform over ref haplotypes + emission at last position)
    init(
        ref_panel.row(m - 1),
        genograph[m - 1],
        hmm_params,
        bprobs.slice_mut(s![m - 1, .., ..]),
    );

    let uniform_frac = 1.0 / (n as f32);

    let mut last_cm_dist = None;

    for i in (0..m - 1).rev() {
        let (mut cur_bprob, prev_bprob) =
            bprobs.multi_slice_mut((s![i, .., ..], s![i + 1, .., ..]));

        if !ignored_sites[i] || genograph[i + 1].is_segment_marker() {
            let rprobs = if let Some(last_cm_dist) = last_cm_dist {
                let cm_dist = last_cm_dist - variants[i].cm;
                hmm_params.get_rprobs_from_cm_dist(cm_dist)
            } else {
                hmm_params.get_backward_rprobs(i)
            };
            last_cm_dist = None;
            if genograph[i + 1].is_segment_marker() {
                collapse(
                    rprobs,
                    uniform_frac.into(),
                    cur_bprob.view_mut(),
                    prev_bprob.view(),
                );
            } else {
                transition(
                    rprobs,
                    uniform_frac.into(),
                    cur_bprob.view_mut(),
                    prev_bprob.view(),
                );
            }
            emission(ref_panel.row(i), genograph[i], hmm_params, cur_bprob);
        } else {
            cur_bprob.assign(&prev_bprob);
            if last_cm_dist.is_none() {
                last_cm_dist = Some(variants[i + 1].cm);
            }
        }
    }

    bprobs
}

// probs: 8 x |cond_haps|
fn init(
    cond_haps: ArrayView1<Genotype>,
    graph_col: G,
    hmm_params: &HmmParamsSlice,
    mut probs: ArrayViewMut2<Real>,
) {
    Zip::indexed(probs.rows_mut()).for_each(|i, mut a| {
        Zip::from(&mut a).and(cond_haps).for_each(|c, &d| {
            *c = emission_prob(d, graph_col.get_row(i), hmm_params.eprob);
        });
    });
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
            1.0
        } else {
            error_prob.0 / error_prob.1
        }
    }
}

fn transition(
    rprob: (Real, Real),
    uniform_frac: Real,
    mut cur_probs: ArrayViewMut2<Real>,
    prev_probs: ArrayView2<Real>,
) {
    let mut sum = 0.;
    Zip::from(cur_probs.rows_mut())
        .and(prev_probs.rows())
        .for_each(|mut cur_p_row, prev_p_row| {
            let sum_h: Real = prev_p_row.sum();
            sum += sum_h;
            Zip::from(&mut cur_p_row)
                .and(&prev_p_row)
                .for_each(|cp, &pp| {
                    *cp = pp * rprob.1 + sum_h * rprob.0 * uniform_frac;
                });
        });

    // renormalize
    cur_probs /= sum;
}

fn collapse(
    rprob: (Real, Real),
    uniform_frac: Real,
    mut cur_probs: ArrayViewMut2<Real>,
    prev_probs: ArrayView2<Real>,
) {
    let sum_k = Array1::<Real>::from_shape_fn(prev_probs.ncols(), |i| prev_probs.column(i).sum());
    let sum = sum_k.sum();
    Zip::from(cur_probs.columns_mut())
        .and(&sum_k)
        .for_each(|mut p_col, s| {
            let p = s / sum * rprob.1 + uniform_frac * rprob.0;
            p_col.fill(p);
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

fn combine(fprobs: ArrayView2<Real>, bprobs: ArrayView2<Real>, mut tprobs: ArrayViewMut2<Real>) {
    Zip::from(tprobs.rows_mut())
        .and(fprobs.rows())
        .for_each(|mut t_row, f_row| {
            Zip::from(&mut t_row)
                .and(bprobs.rows())
                .for_each(|t, b_row| *t = f_row.dot(&b_row));
        });
}

pub fn combine_pairs(tprobs: ArrayView2<Real>, mut tprobs_pairs: ArrayViewMut2<Real>) {
    let tprobs = &tprobs / tprobs.sum();
    for i in 0..P {
        for j in 0..P {
            tprobs_pairs[[i, j]] = tprobs[[i, j]] * tprobs[[P - 1 - i, P - 1 - j]];
        }
    }
    tprobs_pairs /= tprobs_pairs.sum();
}
