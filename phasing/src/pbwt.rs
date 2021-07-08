use ndarray::{Array1, ArrayView2};

#[derive(Clone)]
pub struct PBWTColumn {
    pub a: Array1<u32>,
    pub d: Array1<u32>,
}

pub struct PBWT<'a> {
    prev_col: PBWTColumn,
    x: ArrayView2<'a, u8>,
    m: usize,
    n: usize,
    cur_i: usize,
}

impl<'a> PBWT<'a> {
    pub fn new(x: ArrayView2<'a, u8>) -> Self {
        let n = x.nrows();
        let m = x.ncols();
        let prev_col = PBWTColumn {
            a: Array1::<u32>::from_iter(0..n as u32),
            d: unsafe { Array1::<u32>::uninit(n).assume_init() },
        };
        Self {
            prev_col,
            x,
            m,
            n,
            cur_i: 0,
        }
    }
}

impl<'a> Iterator for PBWT<'a> {
    type Item = (PBWTColumn, u32);
    fn next(&mut self) -> Option<Self::Item> {
        let mut cur_col = PBWTColumn {
            a: unsafe { Array1::<u32>::uninit(self.n).assume_init() },
            d: unsafe { Array1::<u32>::uninit(self.n).assume_init() },
        };

        if self.cur_i >= self.m {
            None
        } else {
            let i = self.cur_i;
            self.cur_i += 1;
            let n_zeros = self.x.row(i).iter().filter(|&&v| v == 0).count();
            // original PBWT
            let mut u = 0;
            let mut v = n_zeros;
            let mut p = i + 1;
            let mut q = i + 1;

            for j in 0..self.n {
                let prev_a = self.prev_col.a[j] as usize;
                let prev_d = self.prev_col.d[j] as usize;
                p = p.max(prev_d);
                q = q.max(prev_d);
                if self.x[[i, prev_a]] == 0 {
                    cur_col.a[u] = prev_a as u32;
                    cur_col.d[u] = p as u32;
                    u += 1;
                    p = 0;
                } else {
                    cur_col.a[v] = prev_a as u32;
                    cur_col.d[v] = q as u32;
                    v += 1;
                    q = 0;
                }
            }
            self.prev_col = cur_col.clone();
            Some((cur_col, n_zeros as u32))
        }
    }
}
