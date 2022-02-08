use crate::pbwt::{PBWTColumn, PBWT};
use crate::{tp_value, Genotype, UInt};
use ndarray::{Array1, ArrayView1};

#[cfg(feature = "leak-resist")]
use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpOrd};

pub fn find_neighbors(
    ref_panel: impl Iterator<Item = Array1<i8>>,
    mut estimated_haps: impl Iterator<Item = Array1<Genotype>>,
    n_pos: usize,
    n_haps_ref: usize,
    n_haps_target: usize,
    s: usize,
) -> Vec<bool> {
    let mut pbwt = PBWT::new(ref_panel, n_pos, n_haps_ref);
    let mut prev_col = pbwt.get_init_col().unwrap();

    let mut prev_target = vec![Target::default(); n_haps_target];

    let mut neighbors_bitmap = vec![false; n_haps_ref];

    for i in 0..n_pos {
        let (cur_col, cur_n_zeros, hap_row) = pbwt.next().unwrap();
        let cur_haps = estimated_haps.next().unwrap();

        for j in 0..n_haps_target {
            let cur_target = find_target_single_marker(
                hap_row.view(),
                cur_n_zeros as u32,
                cur_haps[j],
                i as u32,
                &prev_target[j],
                &prev_col,
            );

            let (new_neighbors, _) =
                find_neighbors_single_marker(i as u32, s, &cur_target, &cur_col, &prev_col, false);

            for i in new_neighbors {
                neighbors_bitmap[i as usize] = true;
            }

            prev_target[j] = cur_target;
        }

        prev_col = cur_col;
    }
    neighbors_bitmap
}

#[derive(Clone)]
pub struct Target {
    pub ind: UInt,
    pub d: UInt,
    pub post_d: UInt,
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

#[cfg(feature = "leak-resist")]
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

    // oblivious PBWT lookup
    let mut target_u = tp_value!(0, u32);
    let mut target_v = tp_value!(cur_n_zeros, u32);
    let mut target_p = tp_value!(i + 1, u32);
    let mut target_q = tp_value!(i + 1, u32);

    for j in 0..n + 1 {
        #[cfg(feature = "leak-resist")]
        {
            let cond1 = prev_target.ind.tp_eq(&(j as u32));
            target_p = cond1.select(
                prev_target
                    .d
                    .tp_gt(&target_p)
                    .select(prev_target.d, target_p),
                target_p,
            );
            target_q = cond1.select(
                prev_target
                    .d
                    .tp_gt(&target_q)
                    .select(prev_target.d, target_q),
                target_q,
            );
            let cond2 = cur_t.tp_eq(&0);
            cur_target.ind = cond1.select(cond2.select(target_u, target_v), cur_target.ind);
            cur_target.d = cond1.select(cond2.select(target_p, target_q), cur_target.d);
            let cond3 = cond2.select(
                cur_target.ind.tp_eq(&cur_n_zeros),
                cur_target.ind.tp_eq(&(n as u32)),
            );
            cur_target.post_d = (cond1 & cond3).select(UInt::protect(i + 1), cur_target.post_d);

            target_p = (cond1 & cond2).select(tp_value!(0, u32), target_p);
            target_u = (cond1 & cond2).select(target_u + 1, target_u);

            target_q = (cond1 & !cond2).select(tp_value!(0, u32), target_q);
            target_v = (cond1 & !cond2).select(target_v + 1, target_v);
        }

        #[cfg(not(feature = "leak-resist"))]
        if j == prev_target.ind as usize {
            target_p = target_p.max(prev_target.d);
            target_q = target_q.max(prev_target.d);
            if cur_t == 0 {
                cur_target.ind = target_u;
                cur_target.d = target_p;
                if cur_target.ind == cur_n_zeros {
                    cur_target.post_d = i + 1;
                }
                target_p = 0;
                target_u += 1;
            } else {
                cur_target.ind = target_v;
                cur_target.d = target_q;
                if cur_target.ind == n as u32 {
                    cur_target.post_d = i + 1;
                }
                target_q = 0;
                target_v += 1;
            }
        }

        if j < n {
            let prev_a = prev_col.a[j];
            let prev_d = prev_col.d[j];

            #[cfg(feature = "leak-resist")]
            {
                let cond1 = prev_target.ind.tp_eq(&(j as u32));
                target_p = cond1.select(
                    target_p
                        .tp_gt(&prev_target.post_d)
                        .select(target_p, prev_target.post_d),
                    target_p,
                );
                target_q = cond1.select(
                    target_q
                        .tp_gt(&prev_target.post_d)
                        .select(target_q, prev_target.post_d),
                    target_q,
                );

                let prev_d = UInt::protect(prev_d as u32);
                target_p =
                    (!cond1).select(target_p.tp_gt(&prev_d).select(target_p, prev_d), target_p);
                target_q =
                    (!cond1).select(target_q.tp_gt(&prev_d).select(target_q, prev_d), target_q);
            }

            #[cfg(not(feature = "leak-resist"))]
            if j == prev_target.ind as usize {
                target_p = target_p.max(prev_target.post_d);
                target_q = target_q.max(prev_target.post_d);
            } else {
                target_p = target_p.max(prev_d);
                target_q = target_q.max(prev_d);
            }

            if cur_x_col[prev_a as usize] == 0 {
                #[cfg(feature = "leak-resist")]
                {
                    let cond1 = target_u.tp_eq(&(cur_target.ind + 1));
                    cur_target.post_d = cond1.select(target_p, cur_target.post_d);
                }

                #[cfg(not(feature = "leak-resist"))]
                if target_u == cur_target.ind + 1 {
                    cur_target.post_d = target_p;
                }

                target_p = tp_value!(0, u32);
                target_u += 1;
            } else {
                #[cfg(feature = "leak-resist")]
                {
                    let cond1 = target_v.tp_eq(&(cur_target.ind + 1));
                    cur_target.post_d = cond1.select(target_q, cur_target.post_d);
                }

                #[cfg(not(feature = "leak-resist"))]
                if target_v == cur_target.ind + 1 {
                    cur_target.post_d = target_q;
                }

                target_q = tp_value!(0, u32);
                target_v += 1;
            }
        }
    }
    cur_target
}

pub fn find_neighbors_single_marker(
    i: u32,
    s: usize,
    cur_target: &Target,
    cur_col: &PBWTColumn,
    prev_col: &PBWTColumn,
    with_divs: bool,
) -> (Vec<UInt>, Option<Vec<UInt>>) {
    let n = prev_col.a.len();

    let capacity = 2 * s;
    let mut saved_a;
    let mut saved_d;
    let pre_min;
    let post_max;

    #[cfg(feature = "leak-resist")]
    {
        use crate::oram::SmallLSOram;
        saved_a = SmallLSOram::new(capacity);
        saved_d = SmallLSOram::new(capacity);
        let s = UInt::protect(s as u32);
        pre_min = cur_target
            .ind
            .tp_gt_eq(&s)
            .select(tp_value!(0, u32), s - cur_target.ind);
        post_max = (cur_target.ind + s)
            .tp_lt_eq(&(n as u32 - 1))
            .select(2 * s - 1, s + n as u32 - 1 - cur_target.ind);
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        saved_a = vec![0; capacity];
        saved_d = vec![0; capacity];
        pre_min = if cur_target.ind as usize >= s {
            0
        } else {
            s as u32 - cur_target.ind
        };
        post_max = if cur_target.ind as usize + s <= n - 1 {
            2 * s as u32 - 1
        } else {
            n as u32 + s as u32 - 1 - cur_target.ind
        };
    }

    for j in 0..n {
        #[cfg(feature = "leak-resist")]
        {
            let k = UInt::protect(j as u32);
            let s = UInt::protect(s as u32);
            let cond1 = cur_target.ind.tp_gt(&k);
            let dif = cond1.select(cur_target.ind - k, k - cur_target.ind);
            let cond2 = cond1.select(dif.tp_lt_eq(&s), dif.tp_lt(&s));
            let i = cond1.select(s - dif, s + dif);
            saved_d.cond_obliv_write(UInt::protect(cur_col.d[j]), i, cond2);
            saved_a.cond_obliv_write(UInt::protect(cur_col.a[j]), i, cond2);
        }

        #[cfg(not(feature = "leak-resist"))]
        if j < cur_target.ind as usize {
            let dif = cur_target.ind as usize - j;
            if dif <= s {
                saved_d[s - dif] = cur_col.d[j];
                saved_a[s - dif] = cur_col.a[j];
            }
        } else {
            let dif = j - cur_target.ind as usize;
            if dif < s {
                saved_d[s + dif] = cur_col.d[j];
                saved_a[s + dif] = cur_col.a[j];
            }
        }
    }

    // neighbor finding
    let mut pre_ind;
    let mut post_ind;
    let mut pre_div = cur_target.d;
    let mut post_div = cur_target.post_d;

    pre_ind = tp_value!(s as u32 - 1, u32);
    post_ind = tp_value!(s, u32);

    let mut new_neighbors = Vec::with_capacity(s);
    let mut new_neighbor_divs = if with_divs {
        Some(Vec::with_capacity(s))
    } else {
        None
    };
    for _ in 0..s {
        #[cfg(feature = "leak-resist")]
        {
            let chosen_post =
                pre_ind.tp_lt(&pre_min) | post_ind.tp_lt_eq(&post_max) & post_div.tp_lt(&pre_div);
            let mut ind = chosen_post.select(post_ind, pre_ind);
            new_neighbors.push(saved_a.obliv_read(ind));
            if let Some(new_neighbor_divs) = new_neighbor_divs.as_mut() {
                new_neighbor_divs.push(chosen_post.select(post_div, pre_div));
            }
            let cannot_advance =
                (chosen_post & (ind.tp_eq(&post_max))) | (!chosen_post & (ind.tp_eq(&pre_min)));
            ind = (!cannot_advance).select(chosen_post.select(ind + 1, ind - 1), ind);
            let div = (!cannot_advance).select(
                chosen_post.select(saved_d.obliv_read(ind), saved_d.obliv_read(ind + 1)),
                UInt::protect(i as u32),
            );
            pre_div = chosen_post.select(pre_div, pre_div.tp_gt(&div).select(pre_div, div));
            post_div = chosen_post.select(post_div.tp_gt(&div).select(post_div, div), post_div);
            pre_ind = chosen_post.select(pre_ind, ind);
            post_ind = chosen_post.select(ind, post_ind);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            let chosen_post = (pre_ind < pre_min) || (post_ind <= post_max && post_div < pre_div);
            let mut ind = if chosen_post { post_ind } else { pre_ind };
            new_neighbors.push(saved_a[ind as usize]);
            if let Some(new_neighbor_divs) = new_neighbor_divs.as_mut() {
                new_neighbor_divs.push(if chosen_post { post_div } else { pre_div });
            }

            let cannot_advance =
                (chosen_post && (ind == post_max)) || (!chosen_post && (ind == pre_min));

            ind = if chosen_post && !cannot_advance {
                ind + 1
            } else if !chosen_post && !cannot_advance {
                ind - 1
            } else {
                ind
            };

            let div = if chosen_post && !cannot_advance {
                saved_d[ind as usize]
            } else if !chosen_post && !cannot_advance {
                saved_d[ind as usize + 1]
            } else {
                i
            };

            pre_div = if chosen_post {
                pre_div
            } else {
                pre_div.max(div)
            };

            post_div = if chosen_post {
                post_div.max(div)
            } else {
                post_div
            };

            pre_ind = if chosen_post { pre_ind } else { ind };
            post_ind = if chosen_post { ind } else { post_ind };
        }
    }

    (new_neighbors, new_neighbor_divs)
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::Array2;
    use rand::Rng;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn neighbors_finding_test() {
        let mut rng = rand::thread_rng();

        let nhap = 100;
        let npos = 200;
        let n_targets = 3;
        let s = 5;

        let x_ref = Array2::<i8>::from_shape_fn((npos, nhap), |_| rng.gen_range(0..2)); // ref panel
        let x = (0..npos).map(|i| x_ref.row(i).to_owned());
        let t = Array2::from_shape_fn((n_targets, npos), |_| tp_value!(rng.gen_range(0..2), i8));

        let results = find_neighbors(x, t.view(), npos, nhap, s);

        for (result, single_t) in results.into_iter().zip(t.rows().into_iter()) {
            for (i, r) in result.into_iter().enumerate() {
                #[cfg(feature = "leak-resist")]
                let r = r.into_iter().map(|v| v.expose()).collect::<Vec<_>>();
                let mut r = r.into_iter().collect::<HashSet<_>>();
                let mut check_answer = HashMap::<usize, HashSet<u32>>::new();
                for j in 0..nhap {
                    let mut n_matches = 0;
                    for k in (0..i + 1).rev() {
                        #[cfg(feature = "leak-resist")]
                        let a = single_t[k].expose();

                        #[cfg(not(feature = "leak-resist"))]
                        let a = single_t[k];

                        if a == x_ref[[k, j]] {
                            n_matches += 1;
                        } else {
                            break;
                        }
                    }

                    let entry = check_answer
                        .entry(n_matches)
                        .or_insert_with(|| HashSet::new());
                    entry.insert(j as u32);
                }

                let mut all_lens = check_answer.keys().into_iter().cloned().collect::<Vec<_>>();
                all_lens.sort_unstable_by(|a, b| b.cmp(a));

                for j in all_lens {
                    if r.is_empty() {
                        break;
                    }
                    if !check_answer.contains_key(&j) {
                        continue;
                    }
                    assert!(check_answer[&j].is_subset(&r) || r.is_subset(&check_answer[&j]));
                    r = r.difference(&check_answer[&j]).cloned().collect();
                }
                assert!(r.is_empty());
            }
        }
    }
}
