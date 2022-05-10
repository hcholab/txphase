use ndarray::Array1;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

lazy_static::lazy_static! {
    pub static ref EXPAND_T: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_millis(0)));
}

#[derive(Clone, Debug)]
pub struct PBWTColumn {
    pub a: Array1<u32>,
    pub d: Array1<u32>,
}

pub struct PBWT<R: Iterator<Item = Array1<i8>>> {
    prev_col: PBWTColumn,
    x: R,
    npos: usize,
    nhap: usize,
    cur_i: usize,
}

impl<R> PBWT<R>
where
    R: Iterator<Item = Array1<i8>>,
{
    pub fn new(x: R, npos: usize, nhap: usize) -> Self {
        let prev_col = PBWTColumn {
            a: Array1::<u32>::from_iter(0..nhap as u32),
            d: Array1::<u32>::zeros(nhap),
        };
        Self {
            prev_col,
            x,
            npos,
            nhap,
            cur_i: 0,
        }
    }

    pub fn get_init_col(&mut self) -> Option<PBWTColumn> {
        if self.cur_i == 0 {
            Some(self.prev_col.to_owned())
        } else {
            None
        }
    }
}

impl<R> Iterator for PBWT<R>
where
    R: Iterator<Item = Array1<i8>>,
{
    type Item = (PBWTColumn, u32, Array1<i8>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_i >= self.npos {
            None
        } else {
            let mut cur_col = PBWTColumn {
                a: Array1::<u32>::zeros(self.nhap),
                d: Array1::<u32>::zeros(self.nhap),
            };
            let i = self.cur_i;
            let t = Instant::now();
            let hap_row = self.x.next().unwrap();
            let mut _t = EXPAND_T.lock().unwrap();
            *_t += Instant::now() - t;
            self.cur_i += 1;
            let n_zeros = hap_row.iter().filter(|&&v| v == 0).count();
            // original PBWT
            let mut u = 0;
            let mut v = n_zeros;
            let mut p = i + 1;
            let mut q = i + 1;

            for j in 0..self.nhap {
                let prev_a = self.prev_col.a[j] as usize;
                let prev_d = self.prev_col.d[j] as usize;
                p = p.max(prev_d);
                q = q.max(prev_d);
                if hap_row[prev_a] == 0 {
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
            Some((cur_col, n_zeros as u32, hap_row))
        }
    }
}
