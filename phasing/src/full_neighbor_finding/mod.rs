mod pbwt;
use crate::{tp_value, Bool, Genotype, U32};
use ndarray::{Array1, ArrayView1};
use pbwt::{PBWTColumn, PBWT};

#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpOrd};

pub fn find_top_neighbors(
    input_hap_1: &[Bool],
    input_hap_2: &[Bool],
    n_neighbors: usize,
    ref_panel: impl Iterator<Item = Array1<i8>>,
    n_haps: usize,
    find_neighbors_filter: &[bool],
) -> Vec<Option<Vec<U32>>> {
    let n_pos = input_hap_1.len();

    let mut input_hap_1 = input_hap_1.iter().cloned().map(|v| v.as_i8());
    let mut input_hap_2 = input_hap_2.iter().cloned().map(|v| v.as_i8());

    let mut pbwt = PBWT::new(ref_panel, n_pos, n_haps);
    let mut prev_col = pbwt.get_init_col().unwrap();

    let mut prev_target_1 = Target::default();
    let mut prev_target_2 = Target::default();

    let mut all_neighbors = Vec::new();

    for (i, &b) in (0..n_pos).zip(find_neighbors_filter.iter()) {
        let (cur_col, cur_n_zeros, hap_row) = pbwt.next().unwrap();
        let cur_hap_1 = input_hap_1.next().unwrap();
        let cur_hap_2 = input_hap_2.next().unwrap();

        let cur_target_1 = find_target_single_marker(
            hap_row.view(),
            cur_n_zeros as u32,
            cur_hap_1,
            i as u32,
            &prev_target_1,
            &prev_col,
        );

        let cur_target_2 = find_target_single_marker(
            hap_row.view(),
            cur_n_zeros as u32,
            cur_hap_2,
            i as u32,
            &prev_target_2,
            &prev_col,
        );

        if b {
            let nn_1 = cur_target_1
                .d
                .tp_gt(&cur_target_1.post_d)
                .select(tp_value!((i - 1), u32), tp_value!(i, u32));

            let nn_2 = cur_target_2
                .d
                .tp_gt(&cur_target_2.post_d)
                .select(tp_value!((i - 1), u32), tp_value!(i, u32));

            let mut neighbors_1 = vec![tp_value!(0, u32); n_neighbors];
            let mut neighbors_2 = vec![tp_value!(0, u32); n_neighbors];

            for j in 0..n_haps {
                let cond_1 = nn_1.tp_eq(&(j as u32));
                let cond_2 = nn_2.tp_eq(&(j as u32));
                let neighbors_ = find_top_neighbors_pos(j, n_neighbors, &cur_col);

                for (&src, tar) in neighbors_.iter().zip(neighbors_1.iter_mut()) {
                    *tar = cond_1.select(src, *tar);
                }
                for (&src, tar) in neighbors_.iter().zip(neighbors_2.iter_mut()) {
                    *tar = cond_2.select(src, *tar);
                }
            }

            neighbors_1.append(&mut neighbors_2);
            all_neighbors.push(Some(neighbors_1));
        } else {
            all_neighbors.push(None);
        }
        prev_target_1 = cur_target_1;
        prev_target_2 = cur_target_2;
        prev_col = cur_col;
    }
    all_neighbors
}

#[derive(Clone)]
pub struct Target {
    pub ind: U32,
    pub d: U32,
    pub post_d: U32,
}

impl Default for Target {
    fn default() -> Self {
        Self {
            ind: tp_value!(0, u32),
            d: tp_value!(0, u32),
            post_d: tp_value!(0, u32),
        }
    }
}

#[cfg(feature = "obliv")]
impl tp_fixedpoint::timing_shield::TpCondSwap for Target {
    fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
        cond.cond_swap(&mut a.ind, &mut b.ind);
        cond.cond_swap(&mut a.d, &mut b.d);
        cond.cond_swap(&mut a.post_d, &mut b.post_d);
    }
}

pub fn find_target_single_marker(
    cur_x_col: ArrayView1<i8>,
    cur_n_zeros: u32,
    cur_t: Genotype,
    i: u32,
    prev_target: &Target,
    prev_col: &PBWTColumn,
) -> Target {
    let n = prev_col.a.len();
    let mut cur_target = Target::default();

    #[cfg(feature = "obliv")]
    let mut target_u = cur_t
        .tp_eq(&0)
        .select(tp_value!(0, u32), tp_value!(cur_n_zeros, u32));

    #[cfg(not(feature = "obliv"))]
    let mut target_u = if cur_t == 0 { 0 } else { cur_n_zeros };

    let mut target_p = tp_value!(i + 1, u32);
    cur_target.post_d = tp_value!(i + 1, u32);

    #[cfg(feature = "obliv")]
    let mut cont = tp_value!(false, bool);

    for j in 0..n {
        let cur_x = cur_x_col[prev_col.a[j] as usize];

        #[cfg(feature = "obliv")]
        {
            let condx = cur_t.tp_eq(&cur_x);

            // j < prev_target.ind
            let cond1 = prev_target.ind.tp_gt(&(j as u32));
            target_u = (cont & cond1 & condx).select(target_u + 1, target_u);
            target_p = (cont & cond1).select(
                condx.select(
                    tp_value!(0, u32),
                    target_p
                        .tp_gt(&prev_col.d[j])
                        .select(target_p, tp_value!(prev_col.d[j], u32)),
                ),
                target_p,
            );

            // j == prev_target.ind
            let cond2 = prev_target.ind.tp_eq(&(j as u32));
            cur_target.d = (cont & cond2).select(
                prev_target
                    .d
                    .tp_gt(&target_p)
                    .select(prev_target.d, target_p),
                cur_target.d,
            );
            cur_target.ind = (cont & cond2).select(target_u, cur_target.ind);
            target_p = (cont & cond2).select(prev_target.post_d, target_p);
            cur_target.post_d = (cont & cond2 & condx).select(target_p, cur_target.post_d);
            cont = (cont & cond2 & condx).select(tp_value!(false, bool), cont);

            // j > prev_target.ind
            let cond3 = prev_target.ind.tp_lt(&(j as u32));
            target_p = (cont & cond3).select(
                target_p
                    .tp_gt(&prev_col.d[j])
                    .select(target_p, tp_value!(prev_col.d[j], u32)),
                target_p,
            );
            cur_target.post_d = (cont & cond3 & condx).select(target_p, cur_target.post_d);
            cont = (cont & cond3 & condx).select(tp_value!(false, bool), cont);
        }

        #[cfg(not(feature = "obliv"))]
        if j < prev_target.ind as usize {
            if cur_t == cur_x {
                target_u += 1;
                target_p = 0;
            } else {
                target_p = target_p.max(prev_col.d[j]);
            }
        } else if j == prev_target.ind as usize {
            cur_target.d = prev_target.d.max(target_p);
            cur_target.ind = target_u;

            target_p = prev_target.post_d;
            if cur_t == cur_x {
                cur_target.post_d = target_p;
                break;
            }
        } else {
            target_p = target_p.max(prev_col.d[j]);
            if cur_t == cur_x {
                cur_target.post_d = target_p;
                break;
            }
        }
    }

    {
        let cond = prev_target.ind.tp_eq(&(n as u32));
        cur_target.d = cond.select(
            prev_target
                .d
                .tp_gt(&target_p)
                .select(prev_target.d, target_p),
            cur_target.d,
        );
        cur_target.ind = cond.select(target_u, cur_target.ind);
        cur_target.post_d = cond.select(tp_value!(i + 1, u32), cur_target.post_d);
    }

    #[cfg(not(feature = "obliv"))]
    if prev_target.ind == n as u32 {
        cur_target.d = prev_target.d.max(target_p);
        cur_target.ind = target_u;
        cur_target.post_d = i + 1;
    }

    cur_target
}

pub fn find_top_neighbors_pos(pos: usize, n_neighbors: usize, pbwt: &PBWTColumn) -> Vec<U32> {
    let mut neighbors = Vec::with_capacity(n_neighbors);

    let mut pos_up_iter = (0..pos).rev();
    let mut pos_down_iter = pos + 1..pbwt.d.len();
    let mut pos_up = pos_up_iter.next();
    let mut pos_down = pos_down_iter.next();

    let mut div_up = 0;
    let mut div_down = 0;

    for _ in 0..n_neighbors {
        match (pos_up, pos_down) {
            (Some(pos_up_), Some(pos_down_)) => {
                div_up = div_up.max(pbwt.d[pos_up_ + 1]);
                div_down = div_down.max(pbwt.d[pos_down_]);
                if div_up < div_down {
                    neighbors.push(tp_value!(pbwt.a[pos_up_], u32));
                    pos_up = pos_up_iter.next();
                } else {
                    neighbors.push(tp_value!(pbwt.a[pos_down_], u32));
                    pos_down = pos_down_iter.next();
                }
            }
            (Some(pos_up_), None) => {
                neighbors.push(tp_value!(pbwt.a[pos_up_], u32));
                pos_up = pos_up_iter.next();
            }
            (None, Some(pos_down_)) => {
                neighbors.push(tp_value!(pbwt.a[pos_down_], u32));
                pos_down = pos_down_iter.next();
            }
            _ => break,
        };
    }
    neighbors
}
