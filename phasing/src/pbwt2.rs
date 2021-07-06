use ndarray::{Array1, Array2};
use rand::Rng;
use std::mem::swap;
use std::time::Instant;

#[cfg(feature = "leak-resist")]
mod inner {
    pub use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpOrd};
    use tp_fixedpoint::timing_shield::{TpU32, TpU8};
    pub type Genotype = TpU8;
    pub type UInt = TpU32;
}

#[cfg(not(feature = "leak-resist"))]
mod inner {
    pub type Genotype = u8;
    pub type UInt = u32;
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

use inner::*;

fn pbwt(x: &[Vec<u8>], t: &[Genotype], s: usize) -> Vec<Vec<UInt>> {
    let m = x.len();
    let n = x[0].len();

    let x = Array2::from_shape_vec((m, n), x.iter().flatten().cloned().collect()).unwrap();
    let t = Array1::from_vec(t.to_vec());
    let start = Instant::now();

    let mut prev_a_col = Array1::<u32>::from_iter(0..n as u32);
    let mut cur_a_col = unsafe { Array1::<u32>::uninit(n).assume_init() };
    let mut prev_d_col = Array1::<u32>::zeros(n);
    let mut cur_d_col = unsafe { Array1::<u32>::uninit(n).assume_init() };

    let mut cur_target_ind;
    let mut prev_target_ind;
    let mut cur_target_d;
    let mut prev_target_d;
    let mut prev_post_target_d;
    let mut cur_post_target_d;
    let mut neighbors: Vec<Vec<UInt>> = Vec::with_capacity(m);

    #[cfg(feature = "leak-resist")]
    {
        cur_target_ind = UInt::protect(0);
        prev_target_ind = UInt::protect(0);
        cur_target_d = UInt::protect(0);
        prev_target_d = UInt::protect(0);
        cur_post_target_d = UInt::protect(0);
        prev_post_target_d = UInt::protect(0);
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        cur_target_ind = 0;
        prev_target_ind = 0;
        cur_target_d = 0;
        prev_target_d = 0;
        cur_post_target_d = 0;
        prev_post_target_d = 0;
    }

    for i in 0..m {
        let n_zeros = x.row(i).iter().filter(|&&v| v == 0).count();
        let mut u = 0;
        let mut v = n_zeros;

        let mut p = i + 1;
        let mut q = i + 1;

        let mut target_p;
        let mut target_q;
        let mut post_target_flag;

        #[cfg(feature = "leak-resist")]
        {
            target_p = UInt::protect(0);
            target_q = UInt::protect(0);
            post_target_flag = TpBool::protect(false);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            target_p = 0;
            target_q = 0;
            post_target_flag = false;
        }

        for j in 0..n + 1 {
            #[cfg(feature = "leak-resist")]
            {
                let cond1 = prev_target_ind.tp_eq(&(j as u32));
                let p = UInt::protect(p as u32);
                let q = UInt::protect(q as u32);
                target_p = cond1.select(prev_target_d.tp_gt(&p).select(prev_target_d, p), target_p);
                target_q = cond1.select(prev_target_d.tp_gt(&q).select(prev_target_d, q), target_q);
                post_target_flag = cond1.select(TpBool::protect(true), post_target_flag);
                let cond2 = t[i].tp_eq(&0);
                cur_target_ind = cond1.select(
                    cond2.select(UInt::protect(u as u32), UInt::protect(v as u32)),
                    cur_target_ind,
                );
                cur_target_d = cond1.select(cond2.select(target_p, target_q), cur_target_d);
                let cond3 = cond2.select(
                    cur_target_ind.tp_eq(&(n_zeros as u32)),
                    cur_target_ind.tp_eq(&(n as u32)),
                );
                cur_post_target_d =
                    (cond1 & cond3).select(UInt::protect(i as u32 + 1), cur_post_target_d);
                post_target_flag = (cond1 & cond3).select(TpBool::protect(false), post_target_flag);
                target_p = (cond1 & cond2).select(UInt::protect(0), target_p);
                target_q = (cond1 & !cond2).select(UInt::protect(0), target_q);
            }

            #[cfg(not(feature = "leak-resist"))]
            if j == prev_target_ind {
                target_p = p.max(prev_target_d);
                target_q = q.max(prev_target_d);
                post_target_flag = true;
                if t[i] == 0 {
                    cur_target_ind = u;
                    cur_target_d = target_p;
                    if cur_target_ind == n_zeros {
                        cur_post_target_d = i + 1;
                        post_target_flag = false;
                    }
                    target_p = 0;
                } else {
                    cur_target_ind = v;
                    cur_target_d = target_q;
                    if cur_target_ind == n {
                        cur_post_target_d = i + 1;
                        post_target_flag = false;
                    }
                    target_q = 0;
                }
            }

            if j < n {
                let prev_a = prev_a_col[j] as usize;
                let prev_d = prev_d_col[j] as usize;
                p = p.max(prev_d);
                q = q.max(prev_d);

                #[cfg(feature = "leak-resist")]
                {
                    let cond1 = prev_target_ind.tp_eq(&(j as u32));
                    target_p = cond1.select(
                        target_p
                            .tp_gt(&prev_post_target_d)
                            .select(target_p, prev_post_target_d),
                        target_p,
                    );
                    target_q = cond1.select(
                        target_q
                            .tp_gt(&prev_post_target_d)
                            .select(target_q, prev_post_target_d),
                        target_q,
                    );

                    let prev_d = UInt::protect(prev_d as u32);
                    target_p = (!cond1 & post_target_flag)
                        .select(target_p.tp_gt(&prev_d).select(target_p, prev_d), target_p);
                    target_q = (!cond1 & post_target_flag)
                        .select(target_q.tp_gt(&prev_d).select(target_q, prev_d), target_q);
                }

                #[cfg(not(feature = "leak-resist"))]
                if j == prev_target_ind {
                    target_p = target_p.max(prev_post_target_d);
                    target_q = target_q.max(prev_post_target_d);
                } else if post_target_flag {
                    target_p = target_p.max(prev_d);
                    target_q = target_q.max(prev_d);
                }

                if x[[i, prev_a]] == 0 {
                    cur_a_col[u] = prev_a as u32;
                    cur_d_col[u] = p as u32;

                    #[cfg(feature = "leak-resist")]
                    {
                        let cond1 = post_target_flag & t[i].tp_eq(&0);
                        cur_post_target_d = cond1.select(target_p, cur_post_target_d);
                        post_target_flag = cond1.select(TpBool::protect(false), post_target_flag);
                    }

                    #[cfg(not(feature = "leak-resist"))]
                    if post_target_flag && t[i] == 0 {
                        cur_post_target_d = target_p;
                        post_target_flag = false;
                    }

                    u += 1;
                    p = 0;
                } else {
                    cur_a_col[v] = prev_a as u32;
                    cur_d_col[v] = q as u32;

                    #[cfg(feature = "leak-resist")]
                    {
                        let cond1 = post_target_flag & t[i].tp_eq(&1);
                        cur_post_target_d = cond1.select(target_q, cur_post_target_d);
                        post_target_flag = cond1.select(TpBool::protect(false), post_target_flag);
                    }

                    #[cfg(not(feature = "leak-resist"))]
                    if post_target_flag && t[i] == 1 {
                        cur_post_target_d = target_q;
                        post_target_flag = false;
                    }

                    v += 1;
                    q = 0;
                }
            }
        }

        let capacity = 2 * s;
        let mut saved_d = SmallORAM::<UInt>::with_capacity(capacity);
        let mut saved_a = SmallORAM::<UInt>::with_capacity(capacity);
        let pre_min;
        let post_max;

        #[cfg(feature = "leak-resist")]
        {
            let s = UInt::protect(s as u32);
            pre_min = cur_target_ind
                .tp_gt_eq(&s)
                .select(UInt::protect(0), s - cur_target_ind);
            post_max = (cur_target_ind + s)
                .tp_lt_eq(&(n as u32 - 1))
                .select(2 * s - 1, s + n as u32 - 1 - cur_target_ind);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            pre_min = if cur_target_ind >= s {
                0
            } else {
                s - cur_target_ind
            };
            post_max = if cur_target_ind + s <= n - 1 {
                2 * s - 1
            } else {
                n + s - 1 - cur_target_ind
            };
        }

        for j in 0..n {
            #[cfg(feature = "leak-resist")]
            {
                let k = UInt::protect(j as u32);
                let s = UInt::protect(s as u32);
                let capacity = UInt::protect(capacity as u32);
                let cond1 = cur_target_ind.tp_gt(&k);
                let dif = cond1.select(cur_target_ind - k, k - cur_target_ind);
                let cond2 = cond1.select(dif.tp_lt_eq(&s), dif.tp_lt(&s));
                let i = cond2.select(cond1.select(s - dif, s + dif), capacity);
                saved_d.write(UInt::protect(cur_d_col[j]), i);
                saved_a.write(UInt::protect(cur_a_col[j]), i);
            }

            #[cfg(not(feature = "leak-resist"))]
            if j < cur_target_ind {
                let dif = cur_target_ind - j;
                if dif <= s {
                    saved_d.write(cur_d_col[j], (s - dif) as u32);
                    saved_a.write(cur_a_col[j], (s - dif) as u32);
                }
            } else {
                let dif = j - cur_target_ind;
                if dif < s {
                    saved_d.write(cur_d_col[j], (s + dif) as u32);
                    saved_a.write(cur_a_col[j], (s + dif) as u32);
                }
            }
        }

        // neighbor finding
        let mut pre_ind;
        let mut post_ind;
        let mut pre_div = cur_target_d;
        let mut post_div = cur_post_target_d;

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
                let chosen_post = pre_ind.tp_lt(&pre_min)
                    | post_ind.tp_lt_eq(&post_max) & post_div.tp_lt(&pre_div);
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
                let chosen_post =
                    (pre_ind < pre_min) || (post_ind <= post_max && post_div < pre_div);
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
                    saved_d.read(ind as u32) as usize
                } else if !chosen_post && !cannot_advance {
                    saved_d.read(ind as u32 + 1) as usize
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
        neighbors.push(new_neighbors);

        // next round
        prev_target_ind = cur_target_ind;
        prev_target_d = cur_target_d;
        prev_post_target_d = cur_post_target_d;
        swap(&mut prev_a_col, &mut cur_a_col);
        swap(&mut prev_d_col, &mut cur_d_col);
    }

    println!("time = {} ms", (Instant::now() - start).as_millis());
    neighbors
}

#[cfg(test)]
mod test {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn pbwt_lookup() {
        let mut rng = rand::thread_rng();

        let n = 100;
        let m = 100;
        let s = 4;

        let mut x = vec![vec![0 as u8; n]; m]; // ref panel
        let mut t = vec![0 as u8; m]; // target

        for i in 0..m {
            for j in 0..n {
                x[i][j] = rng.gen_range(0..2);
            }
            t[i] = rng.gen_range(0..2);
        }

        #[cfg(feature = "leak-resist")]
        let t = t
            .into_iter()
            .map(|v| Genotype::protect(v))
            .collect::<Vec<_>>();

        let results = pbwt(&x, &t, s);

        for (i, r) in (0..m).zip(results.into_iter()) {
            #[cfg(feature = "leak-resist")]
            let r = r.into_iter().map(|v| v.expose()).collect::<Vec<_>>();

            let mut r = r.into_iter().collect::<HashSet<_>>();
            let mut check_answer = HashMap::<usize, HashSet<u32>>::new();
            let suffix_t = &t[..i + 1];
            let suffix_x = &x[..i + 1];
            for i in 0..n {
                let mut n_matches = 0;
                for (&a, b) in suffix_t.iter().rev().zip(suffix_x.iter().rev()) {
                    #[cfg(feature = "leak-resist")]
                    let a = a.expose();
                    if a == b[i] {
                        n_matches += 1;
                    } else {
                        break;
                    }
                }

                let entry = check_answer
                    .entry(n_matches)
                    .or_insert_with(|| HashSet::new());
                entry.insert(i as u32);
            }

            println!("{:?}", r);
            println!("{:?}", check_answer);

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
