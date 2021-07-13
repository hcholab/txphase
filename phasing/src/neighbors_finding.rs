use crate::pbwt::{PBWTColumn, PBWT};
use ndarray::{ArrayView1, ArrayView2};
use std::time::Instant;
use crate::{Genotype, UInt};

#[cfg(feature = "leak-resist")]
mod inner {
    use super::*;
    pub use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpOrd};
    pub const ZERO: UInt = unsafe { std::mem::transmute(0u32) };
}

#[cfg(not(feature = "leak-resist"))]
mod inner {
    pub const ZERO: u32 = 0;
}

use inner::*;

pub fn find_neighbors(x: ArrayView2<i8>, t: ArrayView2<Genotype>, s: usize) -> Vec<Vec<Vec<UInt>>> {
    let m = x.nrows();
    let n_targets = t.nrows();

    let start = Instant::now();

    let mut pbwt = PBWT::new(x.view());
    let mut prev_col = pbwt.get_init_col().unwrap();

    let mut prev_target = (0..n_targets)
        .map(|_| Target {
            ind: ZERO,
            d: ZERO,
            post_d: ZERO,
        })
        .collect::<Vec<_>>();

    let mut neighbors: Vec<Vec<Vec<UInt>>> = vec![Vec::with_capacity(m); n_targets];

    for i in 0..m {
        let (cur_col, cur_n_zeros) = pbwt.next().unwrap();

        for j in 0..n_targets {
            let cur_target = find_target_single_marker(
                x.row(i),
                cur_n_zeros as u32,
                t[[j, i]],
                i as u32,
                &prev_target[j],
                &prev_col,
            );

            let new_neighbors =
                find_neighbors_single_marker(i as u32, s, &cur_target, &cur_col, &prev_col);

            prev_target[j] = cur_target;
            neighbors[j].push(new_neighbors);
        }

        prev_col = cur_col;
    }

    println!("time = {} ms", (Instant::now() - start).as_millis());
    neighbors
}

pub struct SmallORAM<T: 'static + Clone> {
    slice_inner: &'static mut [T],
    inner: Box<oram_sgx::align::A64Bytes<64>>,
    capacity: UInt,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static + Clone> SmallORAM<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        assert!((capacity + 1) <= 64 / std::mem::size_of::<T>());
        let mut inner = Box::new(oram_sgx::align::A64Bytes::default());
        let inner_ptr = inner.as_mut_slice() as *mut _ as *mut T;
        let slice_inner = unsafe { std::slice::from_raw_parts_mut(inner_ptr, capacity + 1) };

        let capacity = capacity as u32;
        #[cfg(feature = "leak-resist")]
        let capacity = UInt::protect(capacity);

        Self {
            slice_inner,
            inner,
            capacity,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn write(&mut self, v: T, i: UInt) {
        #[cfg(feature = "leak-resist")]
        let i = i.tp_gt_eq(&self.capacity).select(self.capacity, i).expose();

        #[cfg(not(feature = "leak-resist"))]
        assert!(i < self.capacity);

        self.slice_inner[i as usize] = v;
    }

    pub fn read(&self, i: UInt) -> T {
        #[cfg(feature = "leak-resist")]
        let i = i.tp_gt(&self.capacity).select(self.capacity, i).expose();

        #[cfg(not(feature = "leak-resist"))]
        assert!(i < self.capacity);

        self.slice_inner[i as usize].clone()
    }
}

#[derive(Clone)]
struct Target {
    ind: UInt,
    d: UInt,
    post_d: UInt,
}

fn find_target_single_marker(
    cur_x_col: ArrayView1<i8>,
    cur_n_zeros: u32,
    cur_t: Genotype,
    i: u32,
    prev_target: &Target,
    prev_col: &PBWTColumn,
) -> Target {
    let n = prev_col.a.len();
    let mut cur_target = Target {
        ind: ZERO,
        d: ZERO,
        post_d: ZERO,
    };

    // oblivious PBWT lookup
    let mut target_u = ZERO;
    let mut target_v: UInt;
    let mut target_p: UInt;
    let mut target_q: UInt;

    #[cfg(feature = "leak-resist")]
    {
        target_v = UInt::protect(cur_n_zeros);
        target_p = UInt::protect(i + 1);
        target_q = UInt::protect(i + 1);
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        target_v = cur_n_zeros;
        target_p = i + 1;
        target_q = i + 1;
    }

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

            target_p = (cond1 & cond2).select(ZERO, target_p);
            target_u = (cond1 & cond2).select(target_u + 1, target_u);

            target_q = (cond1 & !cond2).select(ZERO, target_q);
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

                target_p = ZERO;
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

                target_q = ZERO;
                target_v += 1;
            }
        }
    }
    cur_target
}

fn find_neighbors_single_marker(
    i: u32,
    s: usize,
    cur_target: &Target,
    cur_col: &PBWTColumn,
    prev_col: &PBWTColumn,
) -> Vec<UInt> {
    let n = prev_col.a.len();

    let capacity = 2 * s;
    let mut saved_d = SmallORAM::<UInt>::with_capacity(capacity);
    let mut saved_a = SmallORAM::<UInt>::with_capacity(capacity);
    let pre_min;
    let post_max;

    #[cfg(feature = "leak-resist")]
    {
        let s = UInt::protect(s as u32);
        pre_min = cur_target.ind.tp_gt_eq(&s).select(ZERO, s - cur_target.ind);
        post_max = (cur_target.ind + s)
            .tp_lt_eq(&(n as u32 - 1))
            .select(2 * s - 1, s + n as u32 - 1 - cur_target.ind);
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        pre_min = if cur_target.ind as usize >= s {
            0
        } else {
            s - cur_target.ind as usize
        };
        post_max = if cur_target.ind as usize + s <= n - 1 {
            2 * s - 1
        } else {
            n + s - 1 - cur_target.ind as usize
        };
    }

    for j in 0..n {
        #[cfg(feature = "leak-resist")]
        {
            let k = UInt::protect(j as u32);
            let s = UInt::protect(s as u32);
            let capacity = UInt::protect(capacity as u32);
            let cond1 = cur_target.ind.tp_gt(&k);
            let dif = cond1.select(cur_target.ind - k, k - cur_target.ind);
            let cond2 = cond1.select(dif.tp_lt_eq(&s), dif.tp_lt(&s));
            let i = cond2.select(cond1.select(s - dif, s + dif), capacity);
            saved_d.write(UInt::protect(cur_col.d[j]), i);
            saved_a.write(UInt::protect(cur_col.a[j]), i);
        }

        #[cfg(not(feature = "leak-resist"))]
        if j < cur_target.ind as usize {
            let dif = cur_target.ind as usize - j;
            if dif <= s {
                saved_d.write(cur_col.d[j], (s - dif) as u32);
                saved_a.write(cur_col.a[j], (s - dif) as u32);
            }
        } else {
            let dif = j - cur_target.ind as usize;
            if dif < s {
                saved_d.write(cur_col.d[j], (s + dif) as u32);
                saved_a.write(cur_col.a[j], (s + dif) as u32);
            }
        }
    }

    // neighbor finding
    let mut pre_ind;
    let mut post_ind;
    let mut pre_div = cur_target.d;
    let mut post_div = cur_target.post_d;

    #[cfg(feature = "leak-resist")]
    {
        let s = UInt::protect(s as u32);
        pre_ind = s - 1;
        post_ind = s;
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        pre_ind = s - 1;
        post_ind = s;
    }

    let mut new_neighbors = Vec::with_capacity(s);
    for _ in 0..s {
        #[cfg(feature = "leak-resist")]
        {
            let chosen_post =
                pre_ind.tp_lt(&pre_min) | post_ind.tp_lt_eq(&post_max) & post_div.tp_lt(&pre_div);
            let mut ind = chosen_post.select(post_ind, pre_ind);
            new_neighbors.push(saved_a.read(ind));
            let cannot_advance =
                (chosen_post & (ind.tp_eq(&post_max))) | (!chosen_post & (ind.tp_eq(&pre_min)));
            ind = (!cannot_advance).select(chosen_post.select(ind + 1, ind - 1), ind);
            let div = (!cannot_advance).select(
                chosen_post.select(saved_d.read(ind), saved_d.read(ind + 1)),
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
            new_neighbors.push(saved_a.read(ind as u32));

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
                saved_d.read(ind as u32)
            } else if !chosen_post && !cannot_advance {
                saved_d.read(ind as u32 + 1)
            } else {
                i as u32
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

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::Array2;
    use rand::Rng;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn neighbors_finding_test() {
        let mut rng = rand::thread_rng();

        let n = 100;
        let m = 100;
        let n_targets = 3;
        let s = 5;

        let mut x = Array2::<i8>::zeros((m, n)); // ref panel
        let mut t = Array2::<i8>::zeros((n_targets, m)); // ref panel

        for v in x.iter_mut() {
            *v = rng.gen_range(0..2);
        }

        for v in t.iter_mut() {
            *v = rng.gen_range(0..2);
        }

        #[cfg(feature = "leak-resist")]
        let t = {
            let mut _t = unsafe { Array2::<Genotype>::uninit(t.dim()).assume_init() };
            for (a, &b) in _t.iter_mut().zip(t.iter()) {
                *a = Genotype::protect(b);
            }
            _t
        };

        let results = find_neighbors(x.view(), t.view(), s);

        for (result, single_t) in results.into_iter().zip(t.rows().into_iter()) {
            for (i, r) in result.into_iter().enumerate() {
                #[cfg(feature = "leak-resist")]
                let r = r.into_iter().map(|v| v.expose()).collect::<Vec<_>>();
                let mut r = r.into_iter().collect::<HashSet<_>>();
                let mut check_answer = HashMap::<usize, HashSet<u32>>::new();
                for j in 0..n {
                    let mut n_matches = 0;
                    for k in (0..i + 1).rev() {
                        #[cfg(feature = "leak-resist")]
                        let a = single_t[k].expose();

                        #[cfg(not(feature = "leak-resist"))]
                        let a = single_t[k];

                        if a == x[[k, j]] {
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
