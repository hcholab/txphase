use crate::pbwt::{PBWTColumn, PBWT};
use crate::{tp_value, Genotype, Int, Real, UInt};
use ndarray::{Array1, ArrayView1};

#[cfg(feature = "leak-resist")]
use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpOrd};

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

lazy_static::lazy_static! {
    pub static ref PBWT_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
}

pub fn find_top_neighbors(
    input_hap: &[bool],
    s: usize,
    ref_panel: impl Iterator<Item = Array1<i8>>,
    n_haps_ref: usize,
    pbwt_group_filter: &[bool],
) -> Vec<Option<Vec<usize>>> {
    let n_pos = input_hap.len();
    let t = Instant::now();
    let mut input_hap = input_hap.iter().cloned().map(|v| v as i8);
    let mut pbwt = PBWT::new(ref_panel, n_pos, n_haps_ref);
    let mut prev_col = pbwt.get_init_col().unwrap();
    let mut pbwt_group_filter = pbwt_group_filter.iter().cloned();
    let mut cur_pbwt_group_bit = pbwt_group_filter.next();

    let mut prev_target = Target::default();

    //let mut neighbors_bitmap = vec![false; n_haps_ref];
    let mut all_neighbors = Vec::new();

    for i in 0..n_pos {
        let (cur_col, cur_n_zeros, hap_row) = pbwt.next().unwrap();
        let cur_hap = input_hap.next().unwrap();

        let mut new_neighbors: Option<Vec<usize>> = None;

        let cur_target = find_target_single_marker(
            hap_row.view(),
            cur_n_zeros as u32,
            cur_hap,
            i as u32,
            &prev_target,
            &prev_col,
        );

        if let Some(b) = cur_pbwt_group_bit {
            if b {
                let new_neighbors_ =
                    PBWTDepth::build(i as u32, s, &cur_target, &cur_col, &prev_col)
                        .find_neighbors();

                if let Some(new_neighbors) = new_neighbors.as_mut() {
                    new_neighbors.extend(new_neighbors_.into_iter().map(|v| v as usize));
                } else {
                    new_neighbors = Some(new_neighbors_.into_iter().map(|v| v as usize).collect());
                }
            }
        }
        prev_target = cur_target;

        all_neighbors.push(new_neighbors);
        cur_pbwt_group_bit = pbwt_group_filter.next();
        prev_col = cur_col;
    }
    let mut _t = PBWT_T.lock().unwrap();
    *_t += Instant::now() - t;
    all_neighbors
}

pub fn find_top_neighbors_old(
    ref_panel: impl Iterator<Item = Array1<i8>>,
    mut estimated_haps: impl Iterator<Item = Array1<Genotype>>,
    mut pbwt_group_filter: impl Iterator<Item = bool>,
    n_pos: usize,
    n_haps_ref: usize,
    n_haps_target: usize,
    s: usize,
) -> Vec<Option<Vec<u32>>> {
    let t = Instant::now();
    let mut pbwt = PBWT::new(ref_panel, n_pos, n_haps_ref);
    let mut prev_col = pbwt.get_init_col().unwrap();
    let mut cur_pbwt_group_bit = pbwt_group_filter.next();

    let mut prev_target = vec![Target::default(); n_haps_target];

    //let mut neighbors_bitmap = vec![false; n_haps_ref];
    let mut all_neighbors = Vec::new();

    for i in 0..n_pos {
        let (cur_col, cur_n_zeros, hap_row) = pbwt.next().unwrap();
        let cur_haps = estimated_haps.next().unwrap();

        let mut new_neighbors: Option<Vec<u32>> = None;

        for j in 0..n_haps_target {
            let cur_target = find_target_single_marker(
                hap_row.view(),
                cur_n_zeros as u32,
                cur_haps[j],
                i as u32,
                &prev_target[j],
                &prev_col,
            );

            if let Some(b) = cur_pbwt_group_bit {
                if b {
                    let new_neighbors_ =
                        PBWTDepth::build(i as u32, s, &cur_target, &cur_col, &prev_col)
                            .find_neighbors();

                    if let Some(new_neighbors) = new_neighbors.as_mut() {
                        new_neighbors.extend(new_neighbors_.into_iter());
                    } else {
                        new_neighbors = Some(new_neighbors_);
                    }
                }
            }
            prev_target[j] = cur_target;
        }

        all_neighbors.push(new_neighbors);
        cur_pbwt_group_bit = pbwt_group_filter.next();
        prev_col = cur_col;
    }
    let mut _t = PBWT_T.lock().unwrap();
    *_t += Instant::now() - t;
    all_neighbors
    //neighbors_bitmap
}

pub fn find_neighbors_count(
    ref_panel: impl Iterator<Item = Array1<i8>>,
    mut estimated_haps: impl Iterator<Item = Array1<Genotype>>,
    mut pbwt_group_filter: impl Iterator<Item = bool>,
    n_pos: usize,
    n_haps_ref: usize,
    n_haps_target: usize,
    s: usize,
) -> Vec<usize> {
    let mut pbwt = PBWT::new(ref_panel, n_pos, n_haps_ref);
    let mut prev_col = pbwt.get_init_col().unwrap();
    let mut cur_pbwt_group_bit = pbwt_group_filter.next();

    let mut prev_target = vec![Target::default(); n_haps_target];

    let mut neighbors_count = vec![0; n_haps_ref];

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

            if let Some(b) = cur_pbwt_group_bit {
                if b {
                    let new_neighbors =
                        PBWTDepth::build(i as u32, s, &cur_target, &cur_col, &prev_col)
                            .find_neighbors();
                    for i in new_neighbors {
                        neighbors_count[i as usize] += 1;
                    }
                }
            }
            prev_target[j] = cur_target;
        }
        cur_pbwt_group_bit = pbwt_group_filter.next();
        prev_col = cur_col;
    }
    neighbors_count
}

#[derive(Clone, Debug, PartialEq, Eq)]
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

    let mut target_u = if cur_t == 0 { 0 } else { cur_n_zeros };
    let mut target_p = i + 1;
    cur_target.post_d = i + 1;

    for j in 0..n {
        let cur_x = cur_x_col[prev_col.a[j] as usize];
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

    if prev_target.ind == n as u32 {
        cur_target.d = prev_target.d.max(target_p);
        cur_target.ind = target_u;
        cur_target.post_d = i + 1;
    }

    //let mut cur_target_ref = Target::default();
    //// oblivious PBWT lookup
    //let mut target_u = tp_value!(0, u32);
    //let mut target_v = tp_value!(cur_n_zeros, u32);
    //let mut target_p = tp_value!(i + 1, u32);
    //let mut target_q = tp_value!(i + 1, u32);

    //for j in 0..n + 1 {
    //#[cfg(feature = "leak-resist")]
    //{
    //let cond1 = prev_target.ind.tp_eq(&(j as u32));
    //target_p = cond1.select(
    //prev_target
    //.d
    //.tp_gt(&target_p)
    //.select(prev_target.d, target_p),
    //target_p,
    //);
    //target_q = cond1.select(
    //prev_target
    //.d
    //.tp_gt(&target_q)
    //.select(prev_target.d, target_q),
    //target_q,
    //);
    //let cond2 = cur_t.tp_eq(&0);
    //cur_target.ind = cond1.select(cond2.select(target_u, target_v), cur_target.ind);
    //cur_target.d = cond1.select(cond2.select(target_p, target_q), cur_target.d);
    //let cond3 = cond2.select(
    //cur_target.ind.tp_eq(&cur_n_zeros),
    //cur_target.ind.tp_eq(&(n as u32)),
    //);
    //cur_target.post_d = (cond1 & cond3).select(UInt::protect(i + 1), cur_target.post_d);

    //target_p = (cond1 & cond2).select(tp_value!(0, u32), target_p);
    //target_u = (cond1 & cond2).select(target_u + 1, target_u);

    //target_q = (cond1 & !cond2).select(tp_value!(0, u32), target_q);
    //target_v = (cond1 & !cond2).select(target_v + 1, target_v);
    //}

    //#[cfg(not(feature = "leak-resist"))]
    //if j == prev_target.ind as usize {
    //target_p = target_p.max(prev_target.d);
    //target_q = target_q.max(prev_target.d);
    //if cur_t == 0 {
    //cur_target_ref.ind = target_u;
    //cur_target_ref.d = target_p;
    //if cur_target_ref.ind == cur_n_zeros {
    //cur_target_ref.post_d = i + 1;
    //break;
    //}
    //target_p = 0;
    //target_u += 1;
    //} else {
    //cur_target_ref.ind = target_v;
    //cur_target_ref.d = target_q;
    //if cur_target_ref.ind == n as u32 {
    //cur_target_ref.post_d = i + 1;
    //break;
    //}
    //target_q = 0;
    //target_v += 1;
    //}
    //}

    //if j < n {
    //let prev_a = prev_col.a[j];
    //let prev_d = prev_col.d[j];

    //#[cfg(feature = "leak-resist")]
    //{
    //let cond1 = prev_target.ind.tp_eq(&(j as u32));
    //target_p = cond1.select(
    //target_p
    //.tp_gt(&prev_target.post_d)
    //.select(target_p, prev_target.post_d),
    //target_p,
    //);
    //target_q = cond1.select(
    //target_q
    //.tp_gt(&prev_target.post_d)
    //.select(target_q, prev_target.post_d),
    //target_q,
    //);

    //let prev_d = UInt::protect(prev_d as u32);
    //target_p =
    //(!cond1).select(target_p.tp_gt(&prev_d).select(target_p, prev_d), target_p);
    //target_q =
    //(!cond1).select(target_q.tp_gt(&prev_d).select(target_q, prev_d), target_q);
    //}

    //#[cfg(not(feature = "leak-resist"))]
    //if j == prev_target.ind as usize {
    //target_p = target_p.max(prev_target.post_d);
    //target_q = target_q.max(prev_target.post_d);
    //} else {
    //target_p = target_p.max(prev_d);
    //target_q = target_q.max(prev_d);
    //}

    //if cur_x_col[prev_a as usize] == 0 {
    //#[cfg(feature = "leak-resist")]
    //{
    //let cond1 = target_u.tp_eq(&(cur_target.ind + 1));
    //cur_target.post_d = cond1.select(target_p, cur_target.post_d);
    //}

    //#[cfg(not(feature = "leak-resist"))]
    //if target_u == cur_target.ind + 1 {
    //cur_target_ref.post_d = target_p;
    //}

    //target_p = tp_value!(0, u32);
    //target_u += 1;
    //} else {
    //#[cfg(feature = "leak-resist")]
    //{
    //let cond1 = target_v.tp_eq(&(cur_target.ind + 1));
    //cur_target.post_d = cond1.select(target_q, cur_target.post_d);
    //}

    //#[cfg(not(feature = "leak-resist"))]
    //if target_v == cur_target.ind + 1 {
    //cur_target_ref.post_d = target_q;
    //}

    //target_q = tp_value!(0, u32);
    //target_v += 1;
    //}
    //}
    //}
    //println!("{i}");
    //assert_eq!(cur_target, cur_target_ref);
    cur_target
}
#[cfg(feature = "leak-resist")]
pub struct PBWTDepth {
    saved_a: SmallLSOram,
    saved_d: SmallLSOram,
}

#[cfg(not(feature = "leak-resist"))]
pub struct PBWTDepth {
    s: usize,
    saved_a: Vec<u32>,
    saved_d: Vec<u32>,
    is_saved: Vec<bool>,
    pre_min: u32,
    post_max: u32,
    pre_div: u32,
    post_div: u32,
    i: u32,
}

impl PBWTDepth {
    pub fn build(
        i: u32,
        s: usize,
        cur_target: &Target,
        cur_col: &PBWTColumn,
        prev_col: &PBWTColumn,
    ) -> Self {
        let n = prev_col.a.len();
        let pre_div = cur_target.d;
        let post_div = cur_target.post_d;

        let capacity = 2 * s;
        let mut saved_a;
        let mut saved_d;
        let mut is_saved;
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
            is_saved = vec![false; capacity];
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
                    is_saved[s - dif] = true;
                }
            } else {
                let dif = j - cur_target.ind as usize;
                if dif < s {
                    saved_d[s + dif] = cur_col.d[j];
                    saved_a[s + dif] = cur_col.a[j];
                    is_saved[s + dif] = true;
                }
            }
        }

        Self {
            i,
            s,
            saved_a,
            saved_d,
            is_saved,
            pre_min,
            post_max,
            pre_div,
            post_div,
        }
    }

    pub fn score(&self, hap_pos: ArrayView1<Genotype>) -> Int {
        self.saved_a
            .iter()
            .zip(self.is_saved.iter())
            .filter_map(|(&a, &b)| {
                if b {
                    Some(hap_pos[a as usize] as i32 * 2 - 1)
                } else {
                    None
                }
            })
            .sum()
    }

    pub fn score_div(&self, hap_pos: ArrayView1<Genotype>) -> Real {
        self.saved_a
            .iter()
            .zip(self.saved_d.iter())
            .zip(self.is_saved.iter())
            .filter_map(|((&a, &d), &b)| {
                if b {
                    Some((hap_pos[a as usize] * 2 - 1) as f64 * ((self.i + 2 - d) as f64).ln())
                } else {
                    None
                }
            })
            .sum()
    }

    pub fn find_neighbors(self) -> Vec<UInt> {
        let i = self.i;
        let s = self.s;
        let pre_min = self.pre_min;
        let post_max = self.post_max;
        let mut pre_div = self.pre_div;
        let mut post_div = self.post_div;
        let (saved_a, saved_d) = (self.saved_a, self.saved_d);

        let mut pre_ind;
        let mut post_ind;

        pre_ind = tp_value!(s as u32 - 1, u32);
        post_ind = tp_value!(s, u32);

        let mut new_neighbors = Vec::with_capacity(s);

        let mut cannot_advance_pre = pre_ind < pre_min;
        let mut cannot_advance_post = post_ind > post_max;

        for _ in 0..s {
            #[cfg(feature = "leak-resist")]
            {
                let chosen_post = pre_ind.tp_lt(&pre_min)
                    | post_ind.tp_lt_eq(&post_max) & post_div.tp_lt(&pre_div);
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
                let chosen_post =
                    cannot_advance_pre || (!cannot_advance_post && post_div < pre_div);
                let mut ind = if chosen_post { post_ind } else { pre_ind };
                new_neighbors.push(saved_a[ind as usize]);

                cannot_advance_pre |= !chosen_post && (ind == pre_min);
                cannot_advance_post |= chosen_post && (ind == post_max);

                ind = if chosen_post && !cannot_advance_post {
                    ind + 1
                } else if !chosen_post && !cannot_advance_pre {
                    ind - 1
                } else {
                    ind
                };

                let div = if chosen_post && !cannot_advance_post {
                    saved_d[ind as usize]
                } else if !chosen_post && !cannot_advance_pre {
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

        new_neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, ArrayView2};
    use rand::{Rng, SeedableRng};

    fn gen_ref_panel(n_sites: usize, n_haps: usize, mut rng: impl Rng) -> Array2<i8> {
        let mut ref_panel = Array2::<i8>::zeros((n_sites, n_haps));
        rng.fill(ref_panel.as_slice_mut().unwrap());
        ref_panel.map_inplace(|v| *v &= 1);
        ref_panel
    }
    fn gen_hap(n_sites: usize, mut rng: impl Rng) -> Array1<i8> {
        let mut hap = Array1::<i8>::zeros(n_sites);
        rng.fill(hap.as_slice_mut().unwrap());
        hap.map_inplace(|v| *v &= 1);
        hap
    }

    fn ref_find_top_neighbors(
        ref_panel: ArrayView2<i8>,
        hap: ArrayView1<i8>,
        s: usize,
    ) -> Vec<Vec<usize>> {
        let n_sites = hap.len();
        let ref_panel = ref_panel.t();
        let mut ref_panel = ref_panel
            .rows()
            .into_iter()
            .map(|v| v.to_owned())
            .collect::<Vec<_>>();

        let mut map = std::collections::HashMap::<Vec<i8>, usize>::new();
        for (i, h) in ref_panel.iter().enumerate() {
            map.insert(h.as_slice().unwrap().to_owned(), i);
        }

        let mut all_orders = Vec::new();

        for i in 0..n_sites {
            ref_panel.sort_by(|a, b| {
                reverse_prefix_ord(a.as_slice().unwrap(), b.as_slice().unwrap(), i)
            });

            let mut input_pos = ref_panel.len();
            for (j, h) in ref_panel.iter().enumerate() {
                use std::cmp::Ordering;
                match reverse_prefix_ord(hap.as_slice().unwrap(), h.as_slice().unwrap(), i) {
                    Ordering::Equal | Ordering::Less => {
                        input_pos = j;
                        break;
                    }
                    _ => {}
                }
            }
            let mut orders = Vec::new();

            for (j, h) in ref_panel.iter().enumerate() {
                let prefix_match = compute_div(
                    &hap.as_slice().unwrap()[..i + 1],
                    &h.as_slice().unwrap()[..i + 1],
                );
                if j < input_pos {
                    orders.push((prefix_match, input_pos - j, 1u8, map[h.as_slice().unwrap()]));
                } else {
                    orders.push((
                        prefix_match,
                        j - input_pos + 1,
                        0u8,
                        map[h.as_slice().unwrap()],
                    ));
                }
            }
            orders.sort();
            let orders = orders
                .into_iter()
                .take(s)
                .map(|(_, _, _, id)| id)
                .collect::<Vec<_>>();
            all_orders.push(orders)
        }
        all_orders
    }

    #[test]
    fn neighbors() {
        let n_haps = 100;
        let n_sites = 1000;
        let s = 4;

        let seed = rand::thread_rng().gen::<u64>();
        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(seed);
        let ref_panel = gen_ref_panel(n_sites, n_haps, &mut rng);
        let hap = gen_hap(n_sites, &mut rng);
        let pbwt_group_filter = vec![true; n_sites];

        let neighbors = {
            let hap = hap.clone().into_shape((n_sites, 1)).unwrap();
            find_top_neighbors(
                hap.map(|&v| v == 1).as_slice().unwrap(),
                4,
                ref_panel.rows().into_iter().map(|v| v.to_owned()),
                n_haps,
                &pbwt_group_filter,
            )
        };

        let orders = ref_find_top_neighbors(ref_panel.view(), hap.view(), s);
        assert_eq!(
            neighbors
                .into_iter()
                .map(|v| v.unwrap())
                .collect::<Vec<_>>(),
            orders
        );
    }

    pub fn compute_div(a: &[i8], b: &[i8]) -> usize {
        assert_eq!(a.len(), b.len());
        for (i, (&a_, &b_)) in a.iter().zip(b.into_iter()).enumerate().rev() {
            if a_ != b_ {
                return i + 1;
            }
        }
        return 0;
    }
    pub fn compute_rev_prefix_match(a: &[i8], b: &[i8]) -> usize {
        assert_eq!(a.len(), b.len());
        let mut n = 0;
        for (&a_, &b_) in a.iter().zip(b.into_iter()).rev() {
            if a_ != b_ {
                break;
            }
            n += 1;
        }
        return n;
    }

    pub fn reverse_prefix_ord(a: &[i8], b: &[i8], site_i: usize) -> std::cmp::Ordering {
        let mut ord = std::cmp::Ordering::Equal;
        for (i, j) in a
            .iter()
            .take(site_i + 1)
            .zip(b.iter().take(site_i + 1))
            .rev()
        {
            if i != j {
                ord = i.cmp(j);
                break;
            }
        }
        ord
    }
}
