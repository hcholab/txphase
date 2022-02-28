use crate::neighbors_finding::{find_neighbors_single_marker, find_target_single_marker, Target};
use crate::pbwt::PBWT;
use crate::ref_panel::RefPanel;
use crate::{tp_value, Genotype, Int, UInt};
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

    let mut prev_neighbors_0 = Vec::with_capacity(0);
    let mut prev_neighbors_1 = Vec::with_capacity(0);

    let mut prev_divs_0 = Vec::with_capacity(0);
    let mut prev_divs_1 = Vec::with_capacity(0);

    for i in 0..genotypes.len() {
        let (cur_col, cur_n_zeros, hap_row) = pbwt.next().unwrap();

        let (cur_hap_0, cur_hap_1) = if genotypes[i] == 1 {
            if i == 0 {
                (0, 1)
            } else {
                initialize_het_single_marker(
                    hap_row.view(),
                    &prev_neighbors_0,
                    &prev_neighbors_1,
                    &prev_divs_0,
                    &prev_divs_1,
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
                let (n, d) = find_neighbors_single_marker(
                    i as u32,
                    2,
                    &prev_target_0,
                    &cur_col,
                    &prev_col,
                    true,
                );
                prev_neighbors_0 = n;
                prev_divs_0 = d.unwrap();
                let (n, d) = find_neighbors_single_marker(
                    i as u32,
                    2,
                    &prev_target_1,
                    &cur_col,
                    &prev_col,
                    true,
                );
                prev_neighbors_1 = n;
                prev_divs_1 = d.unwrap();
            }
        }

        prev_col = cur_col;
    }

    estm_haplotypes
}

fn initialize_het_single_marker(
    x_row: ArrayView1<i8>,
    neighbors_0: &[UInt],
    neighbors_1: &[UInt],
    divs_0: &[UInt],
    divs_1: &[UInt],
) -> (Genotype, Genotype) {
    let mut s = score_simple(x_row, &neighbors_0, &neighbors_1);

    if s.abs() <= 1 {
        s = score(x_row, &neighbors_0, &divs_0, &neighbors_1, &divs_1);
    }

    #[cfg(feature = "leak-resist")]
    {
        let cond = s0.tp_gt(&s1);
        first = (!cond).as_i8();
        second = cond.as_i8();
        *prev_target_0 = cond.select(cur_target_0_0, cur_target_0_1);
        *prev_target_1 = cond.select(cur_target_1_1, cur_target_1_0);
    }

    #[cfg(not(feature = "leak-resist"))]
    let (first, second) = if s > 0 { (0, 1) } else { (1, 0) };

    return (first, second);
}

//fn initialize_missing_single_target(
//site_pos: usize,
//x_row: ArrayView1<i8>,
//cur_n_zeros: u32,
//prev_target: &mut Target,
//prev_col: &PBWTColumn,
//cur_col: &PBWTColumn,
//) -> Genotype {
//let (s_0, target_0) = score_single(
//tp_value!(0, i8),
//site_pos,
//x_row,
//cur_n_zeros,
//prev_target,
//prev_col,
//cur_col,
//);

//let (s_1, target_1) = score_single(
//tp_value!(1, i8),
//site_pos,
//x_row,
//cur_n_zeros,
//prev_target,
//prev_col,
//cur_col,
//);

//let out_genotype;

//#[cfg(feature = "leak-resist")]
//{
//let cond = (-s_0).tp_gt_eq(&s_1);
//out_genotype = cond.select(tp_value!(0, i8), tp_value!(1, i8));
//*prev_target = cond.select(target_0, target_1);
//}

//#[cfg(not(feature = "leak-resist"))]
//if -s_0 >= s_1 {
//out_genotype = 0;
//*prev_target = target_0;
//} else {
//out_genotype = 1;
//*prev_target = target_1;
//}
//out_genotype
//}
//
fn score_simple_single(x_row: ArrayView1<i8>, neighbors: &[UInt]) -> Int {
    let mut s = tp_value!(0, i32);
    #[cfg(feature = "leak-resist")]
    {
        for (i, &x) in x_row.iter().enumerate() {
            for (&k, &d) in neighbors.iter().zip(divs.iter()) {
                let cond = k.tp_eq(&(i as u32));
                s += cond.select(
                    tp_value!(x as i32 * 2 - 1, i32) * d.as_i32(),
                    tp_value!(0, i32),
                );
            }
        }
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        for &k in neighbors.iter() {
            s += x_row[k as usize] as i32 * 2 - 1;
        }
    }
    s
}

fn score_simple(x_row: ArrayView1<i8>, neighbors_0: &[UInt], neighbors_1: &[UInt]) -> Int {
    let s_0 = score_simple_single(x_row, neighbors_0);
    let s_1 = score_simple_single(x_row, neighbors_1);
    s_1 - s_0
}

fn score_single(x_row: ArrayView1<i8>, neighbors: &[UInt], divs: &[UInt]) -> Int {
    let mut s = tp_value!(0, i32);
    #[cfg(feature = "leak-resist")]
    {
        for (i, &x) in x_row.iter().enumerate() {
            for (&k, &d) in neighbors.iter().zip(divs.iter()) {
                let cond = k.tp_eq(&(i as u32));
                s += cond.select(
                    tp_value!(x as i32 * 2 - 1, i32) * d.as_i32(),
                    tp_value!(0, i32),
                );
            }
        }
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        for (&k, &d) in neighbors.iter().zip(divs.iter()) {
            s += (x_row[k as usize] as i32 * 2 - 1) * (d as f64).log2() as i32;
        }
    }
    s
}

fn score(
    x_row: ArrayView1<i8>,
    neighbors_0: &[UInt],
    divs_0: &[UInt],
    neighbors_1: &[UInt],
    divs_1: &[UInt],
) -> Int {
    let s_0 = score_single(x_row, neighbors_0, divs_0);
    let s_1 = score_single(x_row, neighbors_1, divs_1);
    s_1 - s_0
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
