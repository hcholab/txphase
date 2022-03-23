mod params;
pub use params::*;

use crate::genotype_graph::{G, P};
use crate::variants::Variant;
use crate::{Genotype, Real};

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayViewMut2, Zip};

pub fn combine_dips(tprobs: ArrayView2<Real>, mut tprobs_dips: ArrayViewMut2<Real>) {
    let scale = 1. / tprobs.sum();
    let tprobs = &tprobs * scale;
    for i in 0..P {
        for j in 0..P {
            tprobs_dips[[i, j]] = tprobs[[i, j]] * tprobs[[P - 1 - i, P - 1 - j]];
        }
    }
    let scale = 1. / tprobs_dips.sum();
    tprobs_dips *= scale;
}

pub fn forward_backward(
    ref_panel: ArrayView2<Genotype>,
    genograph: ArrayView1<G>,
    hmm_params: &HmmParams,
    variants: ArrayView1<Variant>,
    ignored_sites: ArrayView1<bool>,
) -> Array3<Real> {
    let m = ref_panel.nrows();
    let k = ref_panel.ncols();

    let bprobs = backward(ref_panel, genograph, hmm_params, variants, ignored_sites);

    let mut tprobs = unsafe { Array3::<Real>::uninit((m, P, P)).assume_init() };

    {
        let init_probs = Array1::<Real>::from_shape_fn(P, |i| bprobs.slice(s![0, i, ..]).sum());
        Zip::from(tprobs.slice_mut(s![0, .., ..]).rows_mut())
            .for_each(|mut r| r.assign(&init_probs));
    }

    let mut prev_fprobs = unsafe { Array2::<Real>::uninit((P, k)).assume_init() };
    let mut cur_fprobs = unsafe { Array2::<Real>::uninit((P, k)).assume_init() };

    init(
        ref_panel.slice(s![0, ..]),
        genograph[0],
        hmm_params,
        prev_fprobs.view_mut(),
    );

    let mut last_cm = variants[0].cm;

    for i in 1..m {
        if !ignored_sites[i] {
            let rprobs = {
                let cm_dist = variants[i].cm - last_cm;
                hmm_params.get_rprobs(cm_dist)
            };

            transition(rprobs, prev_fprobs.view(), cur_fprobs.view_mut());

            if genograph[i].is_segment_marker() {
                combine(
                    cur_fprobs.view(),
                    bprobs.slice(s![i, .., ..]),
                    tprobs.slice_mut(s![i, .., ..]),
                );
                collapse(cur_fprobs.view_mut());
            }

            emission(
                ref_panel.slice(s![i, ..]),
                genograph[i],
                hmm_params,
                cur_fprobs.view_mut(),
            );

            prev_fprobs.assign(&cur_fprobs);
            last_cm = variants[i].cm;
        }
    }

    tprobs
}

fn backward(
    ref_panel: ArrayView2<Genotype>,
    genograph: ArrayView1<G>,
    hmm_params: &HmmParams,
    variants: ArrayView1<Variant>,
    ignored_sites: ArrayView1<bool>,
) -> Array3<Real> {
    let m = ref_panel.nrows();
    let n = ref_panel.ncols();

    let mut bprobs = unsafe { Array3::<Real>::uninit((m, P, n)).assume_init() };

    let mut last_cm = variants[m - 1].cm;

    init(
        ref_panel.row(m - 1),
        genograph[m - 1],
        hmm_params,
        bprobs.slice_mut(s![m - 1, .., ..]),
    );

    for i in (0..m - 1).rev() {
        let (mut cur_bprob, prev_bprob) =
            bprobs.multi_slice_mut((s![i, .., ..], s![i + 1, .., ..]));

        if !ignored_sites[i] || genograph[i + 1].is_segment_marker() {
            let rprobs = {
                let cm_dist = last_cm - variants[i].cm;
                hmm_params.get_rprobs(cm_dist)
            };
            transition(rprobs, prev_bprob.view(), cur_bprob.view_mut());
            if genograph[i + 1].is_segment_marker() {
                collapse(cur_bprob.view_mut());
            }
            emission(ref_panel.row(i), genograph[i], hmm_params, cur_bprob);
            last_cm = variants[i].cm;
        } else {
            cur_bprob.assign(&prev_bprob);
        }
    }
    bprobs
}

fn init(
    cond_haps: ArrayView1<Genotype>,
    graph_col: G,
    hmm_params: &HmmParams,
    mut probs: ArrayViewMut2<Real>,
) {
    Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
        Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
            if z != graph_col.get_row(i) {
                *p = hmm_params.eprob;
            } else {
                *p = 1.;
            }
        });
    });
}

fn emission(
    cond_haps: ArrayView1<Genotype>,
    graph_col: G,
    hmm_params: &HmmParams,
    mut probs: ArrayViewMut2<Real>,
) {
    Zip::indexed(probs.rows_mut()).for_each(|i, mut p_row| {
        Zip::from(&mut p_row).and(cond_haps).for_each(|p, &z| {
            if z != graph_col.get_row(i) {
                *p *= hmm_params.eprob;
            }
        });
    });
}

fn transition(
    rprob: (Real, Real),
    prev_probs: ArrayView2<Real>,
    mut cur_probs: ArrayViewMut2<Real>,
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
                    *cp = pp * rprob.1 + sum_h * rprob.0;
                });
        });

    let scale = 1. / sum;
    cur_probs *= scale;
}

fn collapse(mut cur_probs: ArrayViewMut2<Real>) {
    let mut sum_k = Array1::<Real>::from_shape_fn(cur_probs.ncols(), |i| cur_probs.column(i).sum());
    let scale = 1. / sum_k.sum() as f64;
    sum_k *= scale;
    Zip::from(cur_probs.rows_mut()).for_each(|mut p_row| p_row.assign(&sum_k));
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
