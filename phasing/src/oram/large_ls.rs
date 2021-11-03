use maligned::{align_first, A64};
use timing_shield::{TpBool, TpEq, TpU32};

fn obliv_cmov_a64s(src_a64s: &[A64], dest_a64s: &mut [A64], cond: TpBool) {
    assert!(!src_a64s.is_empty());
    assert!(src_a64s.len() <= dest_a64s.len());
    let n_a64s = src_a64s.len();
    let src_ptr = src_a64s.as_ptr() as *const u64;
    let dest_ptr = dest_a64s.as_mut_ptr() as *mut u64;
    unsafe {
        #[cfg(target_feature = "avx2")]
        super::align::cmov_byte_slice_a64(cond.expose(), src_ptr, dest_ptr, n_a64s * 64);
        #[cfg(not(target_feature = "avx2"))]
        super::align::cmov_byte_slice_a8(cond.expose(), src_ptr, dest_ptr, n_a64s * 8);
    }
}

pub fn a64s_from_bytes_slice(src: &[u8]) -> Vec<A64> {
    let n_a64s = (src.len() + 63) / 64;
    let mut out_block = align_first::<A64, A64>(n_a64s);
    out_block.resize(n_a64s, A64::default());
    let dest_slice =
        unsafe { std::slice::from_raw_parts_mut(out_block.as_mut_ptr() as *mut u8, src.len()) };
    dest_slice.copy_from_slice(src);
    out_block
}

pub fn bytes_slice_from_a64s(src: &[A64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 64) }
}

pub struct LSOramCore {
    inner: Vec<A64>,
    a64s_per_block: usize,
    n_blocks: usize,
}

impl LSOramCore {
    pub fn new(block_size_bytes: usize, n_blocks: usize) -> Self {
        let a64s_per_block = (block_size_bytes + 63) / 64;
        let mut inner = align_first::<A64, A64>(n_blocks * a64s_per_block);
        inner.resize(n_blocks * a64s_per_block, A64::default());
        Self {
            inner,
            a64s_per_block,
            n_blocks,
        }
    }

    pub fn cond_obliv_read(&self, block_index: TpU32, cond: TpBool) -> Vec<A64> {
        let mut out_block = align_first::<A64, A64>(self.a64s_per_block);
        out_block.resize(self.a64s_per_block, A64::default());
        for (i, src_block) in self.inner.chunks_exact(self.a64s_per_block).enumerate() {
            let is_target_block = block_index.tp_eq(&(i as u32)) & cond;
            obliv_cmov_a64s(src_block, out_block.as_mut_slice(), is_target_block);
        }
        out_block
    }

    pub fn cond_oliv_write(&mut self, block_index: TpU32, src_block: &[A64], cond: TpBool) {
        assert!(src_block.len() <= self.a64s_per_block);
        for (i, dest_block) in self.inner.chunks_exact_mut(self.a64s_per_block).enumerate() {
            let is_target_block = block_index.tp_eq(&(i as u32)) & cond;
            obliv_cmov_a64s(src_block, dest_block, is_target_block);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ls_core_rw() {
        let mut ls_core = LSOramCore::new(100, 10);

        let cond = TpBool::protect(true);
        let obliv_index = TpU32::protect(8);
        let blank = ls_core.cond_obliv_read(obliv_index, cond);
        let blank_ref = vec![0u8; 100];
        assert_eq!(&bytes_slice_from_a64s(&blank)[..100], &blank_ref[..]);

        let data_ref = [1, 2, 3, 4, 5, 6, 7, 8u8];
        let data_a64s_ref = a64s_from_bytes_slice(&data_ref);
        ls_core.cond_oliv_write(obliv_index, &data_a64s_ref, cond);
        let read_block = ls_core.cond_obliv_read(obliv_index, cond);
        assert_eq!(&bytes_slice_from_a64s(&read_block)[..8], &data_ref[..]);
    }
}
