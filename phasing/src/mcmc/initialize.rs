use crate::neighbors_finding::{find_target_single_marker, PBWTDepth, Target};
use crate::pbwt::PBWT;
use common::ref_panel::RefPanel;
//use crate::ref_panel::RefPanel;
use crate::{tp_value, Genotype};
use ndarray::{Array2, ArrayView1};

#[cfg(feature = "leak-resist")]
use tp_fixedpoint::timing_shield::{TpEq, TpOrd};

pub fn initialize(ref_panel: &RefPanel, genotypes: ArrayView1<Genotype>) -> Array2<Genotype> {
    let mut pbwt = PBWT::new(ref_panel.iter(), genotypes.len(), ref_panel.n_haps);
    let mut prev_col = pbwt.get_init_col().unwrap();
    let mut prev_target_0 = Target::default();
    let mut prev_target_1 = Target::default();
    let mut estm_haplotypes =
        Array2::<Genotype>::from_elem((genotypes.len(), 2), tp_value!(-1, i8));

    let mut prev_pbwt_depth_0 = None;
    let mut prev_pbwt_depth_1 = None;

    for i in 0..genotypes.len() {
        let (cur_col, cur_n_zeros, hap_row) = pbwt.next().unwrap();

        let (cur_hap_0, cur_hap_1) = if genotypes[i] == 1 {
            if i == 0 {
                (0, 1)
            } else {
                initialize_het_single_marker(
                    hap_row.view(),
                    prev_pbwt_depth_0.as_ref().unwrap(),
                    prev_pbwt_depth_1.as_ref().unwrap(),
                )
            }
        } else {
            (genotypes[i] / 2, genotypes[i] / 2)
        };

        prev_target_0 = find_target_single_marker(
            hap_row.view(),
            cur_n_zeros as u32,
            cur_hap_0,
            i as u32,
            &prev_target_0,
            &prev_col,
        );

        prev_target_1 = find_target_single_marker(
            hap_row.view(),
            cur_n_zeros as u32,
            cur_hap_1,
            i as u32,
            &prev_target_1,
            &prev_col,
        );
        estm_haplotypes[[i, 0]] = cur_hap_0;
        estm_haplotypes[[i, 1]] = cur_hap_1;

        if i + 1 < genotypes.len() {
            if genotypes[i + 1] == 1 {
                prev_pbwt_depth_0 = Some(PBWTDepth::build(
                    i as u32,
                    1,
                    &prev_target_0,
                    &cur_col,
                    &prev_col,
                ));
                prev_pbwt_depth_1 = Some(PBWTDepth::build(
                    i as u32,
                    1,
                    &prev_target_1,
                    &cur_col,
                    &prev_col,
                ));
            }
        }

        prev_col = cur_col;
    }

    estm_haplotypes
}

fn initialize_het_single_marker(
    x_row: ArrayView1<i8>,
    prev_pbwt_depth_0: &PBWTDepth,
    prev_pbwt_depth_1: &PBWTDepth,
) -> (Genotype, Genotype) {
    let mut s = prev_pbwt_depth_0.score(x_row) as i32;
    s -= prev_pbwt_depth_1.score(x_row) as i32;

    if s.abs() > 1 {
        if s > 0 {
            return (1, 0);
        } else {
            return (0, 1);
        }
    }

    let mut s = prev_pbwt_depth_0.score_div(x_row);
    s -= prev_pbwt_depth_1.score_div(x_row);

    if s > 0. {
        return (1, 0);
    } else {
        return (0, 1);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};

    //#[test]
    //fn initialize_test() {
    //let nhap = 100;
    //let npos = 20;

    //let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);
    //let x_ref = ndarray::Array2::<i8>::from_shape_fn((npos, nhap), |_| rng.gen_range(0..2));
    //let x = (0..npos).map(|i| x_ref.row(i).to_owned());
    //let mut t = ndarray::Array1::from_shape_fn(npos, |_| tp_value!(rng.gen_range(0..3), i8));
    //let missing_bitmask = (0..npos)
    //.map(|_| rng.gen_range(0..4) % 4 == 0)
    //.collect::<Vec<_>>();
    //for (&b, v) in missing_bitmask.iter().zip(t.iter_mut()) {
    //if b {
    //*v = tp_value!(-1, i8);
    //}
    //}
    //let result = initialize(x, t.view(), &missing_bitmask[..], npos, nhap);
    //let result_t = &result.row(0) + &result.row(1);

    //#[cfg(feature = "leak-resist")]
    //{
    //use ndarray::Array1;
    //let result = Array2::from_shape_fn(result.dim(), |(i, j)| result[[i, j]].expose());
    //let result_t = Array1::from_shape_fn(result_t.dim(), |i| result_t[i].expose());
    //let t = Array1::from_shape_fn(t.dim(), |i| t[i].expose());

    //println!("{:?}", result);
    //println!("{:?}", result_t);
    //println!("{:?}", t);

    //for ((&b, &ref_v), &v) in missing_bitmask.iter().zip(result_t.iter()).zip(t.iter()) {
    //if !b {
    //assert_eq!(ref_v, v);
    //}
    //}
    //}

    //#[cfg(not(feature = "leak-resist"))]
    //{
    //println!("{:?}", result);
    //println!("{:?}", result_t);
    //println!("{:?}", t);
    //for ((&b, &ref_v), &v) in missing_bitmask.iter().zip(result_t.iter()).zip(t.iter()) {
    //if !b {
    //assert_eq!(ref_v, v);
    //}
    //}
    //}
    //}
}
