use crate::vec::OblivVec;
use timing_shield::{TpBool, TpEq, TpU32, TpU64, TpU8};

pub struct OblivBitmap {
    inner: OblivVec<TpU64>,
    size_bits: usize,
}

impl OblivBitmap {
    pub fn new(size_bits: usize) -> Self {
        Self {
            inner: OblivVec::with_elem(size_bits.div_ceil(64), TpU64::protect(0)),
            size_bits,
        }
    }

    pub fn from_inner(inner: OblivVec<TpU64>, size_bits: usize) -> Self {
        Self { inner, size_bits }
    }

    //pub fn remap_compressed(&self, members: &[Vec<usize>]) -> Vec<TpBool> {
    //let mut compressed_map = vec![TpBool::protect(false); members.len()];
    //for (m, c) in members.into_iter().zip(compressed_map.iter_mut()) {
    //let bit = m
    //.iter()
    //.map(|&i| self.get(i))
    //.reduce(|acc, e| acc | e)
    //.unwrap();
    //*c = bit;
    //}
    //compressed_map
    //}

    pub fn map_from_iter(&mut self, iter: impl Iterator<Item = TpU32>) {
        for i in iter {
            self.set(i);
        }
    }

    pub fn set(&mut self, bit_index: TpU32) {
        let (block_index, inner_index) = cal_ind(bit_index);
        let bitmask = 1u64 << inner_index.expose();
        self.inner.apply(block_index, |x| *x |= bitmask);
    }

    pub fn cond_set(&mut self, bit_index: TpU32, cond: TpBool) {
        let (block_index, inner_index) = cal_ind(bit_index);
        let bitmask = cond.as_u64() << (inner_index.expose() as u32);
        self.inner.apply(block_index, |x| *x |= bitmask);
    }

    pub fn get(&self, bit_index: TpU32) -> TpBool {
        let (block_index, inner_index) = cal_ind(bit_index);
        let bitmask = 1u64 << inner_index.expose();
        (self.inner.get(block_index) & bitmask).tp_not_eq(&0)
    }

    pub fn iter(&self) -> impl Iterator<Item = TpBool> + '_ {
        self.inner
            .iter()
            .cloned()
            .map(|v| (0..64u32).map(move |i| ((v >> i) & 1).tp_eq(&1)))
            .flatten()
            .take(self.size_bits)
    }
}

#[inline]
fn cal_ind(index: TpU32) -> (TpU32, TpU8) {
    (index >> 6, index.as_u8() & 0b111111)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn bitmask() {
        let size = 1000;
        let mut test = OblivBitmap::new(size);
        let mut ref_bitmask = vec![false; size];
        for _ in 0..size {
            let i = rand::thread_rng().gen_range(0..size) as u32;
            test.set(TpU32::protect(i));
            ref_bitmask[i as usize] = true;
            assert!(test.get(TpU32::protect(i)).expose());
        }

        {
            let test = test.iter().map(|v| v.expose()).collect::<Vec<_>>();
            assert_eq!(test, ref_bitmask);
        }

        //let member_size = 5;
        //let members = (0..size / member_size)
        //.map(|i| (i * member_size..(i + 1) * member_size).collect::<Vec<usize>>())
        //.collect::<Vec<_>>();

        //let ref_compressed_bitmask = members
        //.iter()
        //.map(|m| m.iter().fold(false, |acc, &e| acc | ref_bitmask[e]))
        //.collect::<Vec<_>>();

        //let compressed_bitmask = test.remap_compressed(&members);
        //let compressed_bitmask = compressed_bitmask
        //.iter()
        //.map(|v| v.expose())
        //.collect::<Vec<_>>();

        //assert_eq!(compressed_bitmask, ref_compressed_bitmask);
    }
}
