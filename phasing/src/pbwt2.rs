use ndarray::{Array1, Array2};
use rand::Rng;
use std::mem::swap;
use std::time::Instant;

pub fn test_pbwt2() {
    //use rand::SeedableRng;
    //let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
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

    let results = pbwt3(&x, &t, s);

    #[cfg(feature = "leak-resist")]
    let results = {
        let t = t
            .iter()
            .cloned()
            .map(|v| Genotype::protect(v))
            .collect::<Vec<_>>();
        pbwt2(&x, &t, s)
    };

    #[cfg(not(feature = "leak-resist"))]
    let results = pbwt2(&x, &t, s);

    let ref_results = pbwt2_ref(&x, &t, s);

    #[cfg(feature = "leak-resist")]
    let results = results
        .into_iter()
        .map(|vec| vec.into_iter().map(|v| v.expose()).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    assert_eq!(results, ref_results);
}

#[cfg(feature = "leak-resist")]
mod inner {
    pub use tp_fixedpoint::timing_shield::{TpEq, TpOrd};
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
    inner: oram_sgx::align::A64Bytes<64>,
    capacity: UInt,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static + Clone> SmallORAM<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        assert!((capacity + 1) <= 64 / std::mem::size_of::<T>());
        let mut inner = oram_sgx::align::A64Bytes::default();
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

fn pbwt3(x: &[Vec<u8>], t: &[Genotype], s: usize) -> Vec<Vec<UInt>> {
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
    //let mut neighbors: Vec<Vec<u32>> = Vec::with_capacity(m);
    println!("{:?}\n", x);

    {
        cur_target_ind = 0;
        prev_target_ind = 0;
        cur_target_d = 0;
        prev_target_d = 0;
        cur_post_target_d = 0;
        prev_post_target_d = 0;
        //neighbors.push(vec![0; s]);
    }
    println!("cur_a_col = {:?}", prev_a_col);
    println!("cur_d_col = {:?}", prev_d_col);
    println!("");

    for i in 0..m {
        let n_zeros = x.row(i).iter().filter(|&&v| v == 0).count();
        let mut u = 0;
        let mut v = n_zeros;

        let mut p = i + 1;
        let mut q = i + 1;
        let mut target_p = 0;
        let mut target_q = 0;

        let mut post_target_flag = false;

        for j in 0..n + 1 {
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
                    } else {
                        target_p = 0;
                    }
                } else {
                    cur_target_ind = v;
                    cur_target_d = target_q;
                    if cur_target_ind == n {
                        cur_post_target_d = i + 1;
                        post_target_flag = false;
                    } else {
                        target_q = 0;
                    }
                }
            }

            if j < n {
                let prev_a = prev_a_col[j] as usize;
                let prev_d = prev_d_col[j] as usize;
                p = p.max(prev_d);
                q = q.max(prev_d);

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

                    if post_target_flag && t[i] == 0 {
                        cur_post_target_d = target_p;
                        post_target_flag = false;
                    }

                    u += 1;
                    p = 0;
                } else {
                    cur_a_col[v] = prev_a as u32;
                    cur_d_col[v] = q as u32;

                    if post_target_flag && t[i] == 1 {
                        cur_post_target_d = target_q;
                        post_target_flag = false;
                    }

                    v += 1;
                    q = 0;
                }
            }
        }

        use ndarray::s;
        let mut y = Array2::<u8>::zeros((i + 1, n));
        for j in 0..n {
            y.column_mut(j)
                .assign(&x.slice(s![0..i + 1, prev_a_col[j] as usize]));
        }
        println!("prev y = \n{:?}", y);
        for j in 0..n {
            y.column_mut(j)
                .assign(&x.slice(s![0..i + 1, cur_a_col[j] as usize]));
        }
        println!("y = \n{:?}", y);
        println!("cur_a_col = {:?}", cur_a_col);
        println!("cur_d_col = {:?}", cur_d_col);
        println!("");
        for j in 0..n {
            let cur_d = cur_d_col[j] as usize;
            if j == 0 {
                assert_eq!(cur_d, i + 1);
            } else {
                assert_eq!(
                    x.slice(s![cur_d..i, cur_a_col[j - 1] as usize]),
                    x.slice(s![cur_d..i, cur_a_col[j] as usize]),
                );
                if cur_d > 0 && i + 1 - cur_d > 0 {
                    assert_ne!(
                        x.slice(s![(cur_d - 1)..i, cur_a_col[j - 1] as usize]),
                        x.slice(s![(cur_d - 1)..i, cur_a_col[j] as usize]),
                    );
                }
            }
        }

        println!("t = {:?}", t.slice(s![0..i + 1]));
        println!("cur_target_ind = {}", cur_target_ind);
        println!("cur_target_d = {}", cur_target_d);

        if cur_target_ind == 0 {
            assert_eq!(cur_target_d, i + 1);
        } else {
            assert_eq!(
                x.slice(s![cur_target_d..i, cur_a_col[cur_target_ind - 1] as usize]),
                t.slice(s![cur_target_d..i]),
            );
            if cur_target_d > 0 && i + 1 - cur_target_d > 0 {
                assert_ne!(
                    x.slice(s![
                        (cur_target_d - 1)..i,
                        cur_a_col[cur_target_ind - 1] as usize
                    ]),
                    t.slice(s![(cur_target_d - 1)..i,]),
                );
            }
        }

        if cur_target_ind < n {
            println!("cur_post_target_d = {}", cur_post_target_d);
            assert_eq!(
                x.slice(s![cur_post_target_d..i, cur_a_col[cur_target_ind] as usize]),
                t.slice(s![cur_post_target_d..i]),
            );
            if cur_post_target_d > 0 && i + 1 - cur_post_target_d > 0 {
                assert_ne!(
                    x.slice(s![
                        (cur_post_target_d - 1)..i,
                        cur_a_col[cur_target_ind] as usize
                    ]),
                    t.slice(s![(cur_post_target_d - 1)..i,]),
                );
            }
        }

        //use ndarray::s;
        //if cur_target_ind == 0 {
        //assert!(cur_target_d == i);
        //} else {
        //println!("{} {}", i, cur_target_d);
        //assert_eq!(
        //t.slice(s![cur_target_d..(i + 1)]),
        //x.slice(s![cur_target_d..(i + 1), cur_a_col[cur_target_ind - 1] as usize])
        //);
        //}
        //println!("{}", i);
        //assert_eq!(
        //t.slice(s![cur_target_d..(i + 1)]),
        //x.slice(s![cur_target_ind - 1, cur_target_d..(i + 1)])
        //);
        ////if cur_target_d != 0 {
        ////assert_ne!(
        ////&t[(cur_target_d - 1)..(i + 1)],
        ////&x.slice(ndarray::s![(cur_target_d - 1)..(i + 1), target_ind - 1])
        ////.as_slice()
        ////)
        ////}
        //}

        prev_target_ind = cur_target_ind;
        prev_target_d = cur_target_d;
        prev_post_target_d = cur_post_target_d;
        swap(&mut prev_a_col, &mut cur_a_col);
        swap(&mut prev_d_col, &mut cur_d_col);
    }

    todo!()
}

fn pbwt2(x: &[Vec<u8>], t: &[Genotype], s: usize) -> Vec<Vec<UInt>> {
    let m = x.len();
    let n = x[0].len();

    let x = Array2::from_shape_vec((m, n), x.iter().flatten().cloned().collect()).unwrap();
    let t = Array1::from_vec(t.to_vec());

    let start = Instant::now();

    /* PBWT construction WITH TARGET */
    let mut prev_a_col = unsafe { Array1::<u32>::uninit(n).assume_init() };
    let mut cur_a_col = unsafe { Array1::<u32>::uninit(n).assume_init() };
    let mut prev_d_col = unsafe { Array1::<UInt>::uninit(n).assume_init() };
    let mut cur_d_col = unsafe { Array1::<UInt>::uninit(n).assume_init() };

    let mut target_ind;
    let mut target_u;
    let mut target_d;
    let mut neighbors: Vec<Vec<UInt>> = Vec::with_capacity(m);

    // First position special case
    {
        let mut u = 0;
        let mut v = x.row(0).iter().filter(|&&v| v == 0).count();

        let first_x = x.row(0);
        for j in 0..n {
            if first_x[j] == 0 {
                prev_a_col[u] = j as u32;
                u += 1;
            } else {
                prev_a_col[v] = j as u32;
                v += 1;
            }
        }

        #[cfg(feature = "leak-resist")]
        {
            prev_d_col.fill(UInt::protect(0));
        }
        #[cfg(not(feature = "leak-resist"))]
        {
            prev_d_col.fill(0);
        }

        #[cfg(feature = "leak-resist")]
        {
            target_ind = t[0]
                .tp_eq(&1)
                .select(UInt::protect(n as u32), UInt::protect(u as u32)); // target either comes at u or at the end
            target_u = UInt::protect(0);
            target_d = UInt::protect(0);
            neighbors.push(vec![UInt::protect(0); s]);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            target_ind = if t[0] == 1 { n } else { u }; // target either comes at u or at the end
            target_u = 0;
            target_d = 0;
            neighbors.push(vec![0; s]);
        }
    }

    // Second position onwards
    for i in 1..m {
        let n_zeros = x.row(i).iter().filter(|&&v| v == 0).count();
        let mut u = 0usize;
        let mut v = n_zeros;
        let mut p;
        let mut q;
        let t0_flag;

        #[cfg(feature = "leak-resist")]
        {
            p = UInt::protect(i as u32);
            q = UInt::protect(i as u32);
            t0_flag = t[i].tp_eq(&0);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            p = i as u32;
            q = i as u32;
            t0_flag = t[i] == 0;
        }

        for j in 0..n + 1 {
            // Update variables in case target is inserted at this index
            #[cfg(feature = "leak-resist")]
            {
                let at_target_flag = target_ind.tp_eq(&(j as u32));
                p = (at_target_flag & target_d.tp_gt(&p)).select(target_d, p);
                q = (at_target_flag & target_d.tp_gt(&q)).select(target_d, q);
                target_u = at_target_flag.select(UInt::protect(u as u32), target_u);
                target_d = at_target_flag.select(t0_flag.select(p, q), target_d);
                p = (at_target_flag & t0_flag).select(UInt::protect(0), p);
                q = (at_target_flag & !t0_flag).select(UInt::protect(0), q);
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                let at_target_flag = j == target_ind;
                if at_target_flag && (target_d > p as u32) {
                    p = target_d
                }
                if at_target_flag && (target_d > q as u32) {
                    q = target_d
                }
                if at_target_flag {
                    target_u = u;
                }

                if at_target_flag {
                    if t0_flag {
                        target_d = p as u32;
                    } else {
                        target_d = q as u32;
                    }
                }

                if at_target_flag && t0_flag {
                    p = 0;
                }
                if at_target_flag && !t0_flag {
                    q = 0;
                }
            }

            if j < n {
                let aki = prev_a_col[j];
                let dki = prev_d_col[j];

                #[cfg(feature = "leak-resist")]
                {
                    p = dki.tp_gt(&p).select(dki, p);
                    q = dki.tp_gt(&q).select(dki, q);
                }

                #[cfg(not(feature = "leak-resist"))]
                {
                    if dki > p {
                        p = dki;
                    }
                    if dki > q {
                        q = dki;
                    }
                }

                let l;
                let cur_d;

                if x[[i, aki as usize]] == 0 {
                    l = u;
                    u += 1;
                    cur_d = p;
                    #[cfg(feature = "leak-resist")]
                    {
                        p = UInt::protect(0);
                    }
                    #[cfg(not(feature = "leak-resist"))]
                    {
                        p = 0;
                    }
                } else {
                    l = v;
                    v += 1;
                    cur_d = q;
                    #[cfg(feature = "leak-resist")]
                    {
                        q = UInt::protect(0);
                    }
                    #[cfg(not(feature = "leak-resist"))]
                    {
                        q = 0;
                    }
                }

                cur_a_col[l] = aki;
                cur_d_col[l] = cur_d;
            }
        }

        #[cfg(feature = "leak-resist")]
        {
            target_ind = t0_flag.select(
                target_u,
                UInt::protect(n_zeros as u32) + target_ind - target_u,
            );
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            target_ind = if t0_flag {
                target_u
            } else {
                n_zeros + target_ind - target_u
            };
        }

        let mut saved_d = SmallORAM::<UInt>::with_capacity(2 * s);
        let mut saved_a = SmallORAM::<UInt>::with_capacity(2 * s);
        let mut pre_min;
        let mut post_max;

        #[cfg(feature = "leak-resist")]
        {
            pre_min = UInt::protect(s as u32);
            post_max = UInt::protect(s as u32);
            for j in 0..n {
                let cur_d = cur_d_col[j];
                let cur_a = UInt::protect(cur_a_col[j]);

                let j = UInt::protect(j as u32);
                let s = UInt::protect(s as u32);
                let cond1 = target_ind.tp_gt(&j);
                let dif = cond1.select(target_ind - j, j - target_ind);
                let s_m_dif = s - dif;
                let s_p_dif = s + dif;
                let cond2 = cond1.select(dif.tp_lt_eq(&s), dif.tp_lt(&s));
                let ind = cond1.select(cond2.select(s_m_dif, 2 * s), cond2.select(s_p_dif, 2 * s)); // 2 * s for fake writes
                saved_d.write(cur_d, ind);
                saved_a.write(cur_a, ind);
                pre_min = (cond1 & cond2)
                    .select(pre_min.tp_gt(&s_m_dif).select(s_m_dif, pre_min), pre_min);
                post_max = (!cond1 & cond2)
                    .select(post_max.tp_lt(&s_p_dif).select(s_p_dif, post_max), post_max);
            }
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            pre_min = s;
            post_max = s;
            for j in 0..n {
                if target_ind > j {
                    let dif = target_ind - j;
                    if dif <= s {
                        saved_d.write(cur_d_col[j], (s - dif) as u32);
                        saved_a.write(cur_a_col[j], (s - dif) as u32);
                        pre_min = pre_min.min(s - dif);
                    }
                } else {
                    let dif = j - target_ind;
                    if dif < s {
                        saved_d.write(cur_d_col[j], (s + dif) as u32);
                        saved_a.write(cur_a_col[j], (s + dif) as u32);
                        post_max = post_max.max(s + dif);
                    }
                }
            }
        }

        // neighbor finding

        let mut pre_ind;
        let mut post_ind;
        let mut pre_div = target_d;
        let mut post_div;

        #[cfg(feature = "leak-resist")]
        {
            pre_ind = UInt::protect(s as u32 - 1);
            post_ind = UInt::protect(s as u32);
            post_div = (target_ind.tp_eq(&(n as u32)))
                .select(UInt::protect(i as u32), saved_d.read(post_ind));
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            pre_ind = s - 1;
            post_ind = s;
            post_div = if target_ind == n {
                i as u32
            } else {
                saved_d.read(post_ind as u32)
            };
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
        neighbors.push(new_neighbors);

        swap(&mut prev_a_col, &mut cur_a_col);
        swap(&mut prev_d_col, &mut cur_d_col);
    }

    println!("time = {} ms", (Instant::now() - start).as_millis());
    neighbors
}

fn print_marker(shift: usize) {
    for _i in 0..shift {
        print!(" ");
    }
    print!("+\n");
}

fn clip(i: usize, n: usize) -> usize {
    if i >= n {
        return n - 1;
    } else {
        return i;
    }
}
fn min(a: usize, b: usize) -> usize {
    if a < b {
        return a;
    } else {
        return b;
    }
}

fn max(a: usize, b: usize) -> usize {
    if a > b {
        return a;
    } else {
        return b;
    }
}

fn mutex2i(b: bool, v1: i32, v0: i32) -> i32 {
    if b {
        return v1;
    } else {
        return v0;
    }
}

fn mutex2(b: bool, v1: usize, v0: usize) -> usize {
    if b {
        return v1;
    } else {
        return v0;
    }
}

fn mutex4(b1: bool, b2: bool, v11: usize, v10: usize, v01: usize, v00: usize) -> usize {
    if b1 && b2 {
        return v11;
    } else if !b1 && b2 {
        return v01;
    } else if b1 && !b2 {
        return v10;
    } else {
        return v00;
    }
}

fn pbwt2_ref(x: &[Vec<u8>], t: &[u8], s: usize) -> Vec<Vec<u32>> {
    let m = x.len();
    let n = x[0].len();

    let start = Instant::now();

    /* PBWT construction WITH TARGET */
    let mut a_mat = vec![vec![0; n]; m]; // positional prefix arrays
    let mut d_mat = vec![vec![0; n]; m]; // divergence arrays

    let mut a_target = vec![0; m];
    let mut d_target = vec![0; m];

    let mut a = vec![0; n];
    let mut b = vec![0; n];
    let mut d = vec![0; n];
    let mut e = vec![0; n];

    // First position special case
    let mut u = 0;
    let mut v = 0;
    for j in 0..n {
        if x[0][j] == 0 {
            a[u] = j;
            u = u + 1;
        } else {
            b[v] = j;
            v = v + 1;
        }
    }
    for j in 0..u {
        a_mat[0][j] = a[j];
        d_mat[0][j] = 0;
    }
    for j in u..n {
        a_mat[0][j] = b[j - u];
        d_mat[0][j] = 0;
    }

    let mut target_ind = mutex2(t[0] == 1, n, u); // target either comes at u or at the end
    let mut target_u = 0;
    let mut target_d = 0;

    a_target[0] = target_ind;
    d_target[0] = target_d;

    let mut neighbors = vec![vec![0; s]; m];

    // Second position onwards
    for i in 1..m {
        let mut u = 0;
        let mut v = 0;
        let mut p = i;
        let mut q = i;

        let t0_flag = t[i] == 0;

        for j in 0..n + 1 {
            let at_target_flag = j == target_ind;

            // Update variables in case target is inserted at this index
            p = mutex2(at_target_flag && (target_d > p), target_d, p);
            q = mutex2(at_target_flag && (target_d > q), target_d, q);
            target_u = mutex2(at_target_flag, u, target_u);
            target_d = mutex4(at_target_flag, t0_flag, p, q, target_d, target_d);
            p = mutex2(at_target_flag && t0_flag, 0, p);
            q = mutex2(at_target_flag && !t0_flag, 0, q);

            if j < n {
                let aki = a_mat[i - 1][j];
                let dki = d_mat[i - 1][j];
                let x0_flag = x[i][aki] == 0;

                p = mutex2(dki > p, dki, p);
                q = mutex2(dki > q, dki, q);

                if x0_flag {
                    a[u] = aki;
                    d[u] = p;
                    u = u + 1;
                    p = 0;
                } else {
                    b[v] = aki;
                    e[v] = q;
                    v = v + 1;
                    q = 0;
                }
            }
        }
        for j in 0..u {
            a_mat[i][j] = a[j];
            d_mat[i][j] = d[j];
        }
        for j in u..n {
            a_mat[i][j] = b[j - u];
            d_mat[i][j] = e[j - u];
        }

        target_ind = mutex2(t0_flag, target_u, u + target_ind - target_u);
        a_target[i] = target_ind;
        d_target[i] = target_d;

        // neighbor finding
        let mut pre_ind = (target_ind as i32) - 1;
        let mut pre_div = target_d;
        let mut post_ind = target_ind as i32;
        let mut post_div = mutex2(target_ind == n, i, d_mat[i][min(target_ind, n - 1)]);
        for l in 0..s {
            let chosen = (pre_ind < 0) || (post_ind < (n as i32) && post_div < pre_div);
            let mut ind = mutex2i(chosen, post_ind, pre_ind);
            neighbors[i][l] = a_mat[i][ind as usize] as u32;

            ind = mutex2i(chosen, ind + 1, ind - 1);

            // TODO: oblivious lookup for dmat
            let mut div = i;
            div = mutex2(
                chosen && (ind < (n as i32)),
                d_mat[i][clip(ind as usize, n)],
                div,
            );
            div = mutex2(
                !chosen && (ind >= 0),
                d_mat[i][clip((ind + 1) as usize, n)],
                div,
            );

            pre_ind = mutex2i(chosen, pre_ind, ind);
            pre_div = mutex2(chosen, pre_div, max(pre_div, div));
            post_ind = mutex2i(chosen, ind, post_ind);
            post_div = mutex2(chosen, max(post_div, div), post_div);
        }
    }

    println!("ref time = {} ms", (Instant::now() - start).as_millis());

    neighbors

    //[> Output <]
    //for j in 0..n {
    //if (j+1+5 < a_target[m-1]) || (j+1 > a_target[m-1]+5) {
    //continue;
    //}

    //print_marker(d_mat[m-1][j]);

    //for i in 0..m {
    //print!("{}",x[i][a_mat[m-1][j]]);
    //}
    //let mut flag = false;
    //for l in 0..s {
    //if neighbors[m-1][l] == j {
    //flag = true;
    //}
    //}
    //if flag {
    //print!("\tneighbor");
    //}
    //print!("\n");

    //if j+1 == a_target[m-1] {
    //println!("------------");
    //print_marker(d_target[m-1]);
    //for i in 0..m {
    //print!("{}",t[i]);
    //}
    //print!("\ttarget");
    //print!("\n");
    //print_marker(d_mat[m-1][a_target[m-1]]);
    //println!("------------");
    //}
    //}
}
