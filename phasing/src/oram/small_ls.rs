use maligned::{align_first, AsBytes, AsBytesMut, A64};
use std::marker::PhantomData;
use std::mem::size_of;
use timing_shield::{TpBool, TpCondSwap, TpEq, TpU32};

pub struct SmallLSOram<T: 'static + TpCondSwap + Clone> {
    inner: Vec<A64>,
    n: usize,
    elems_per_block: usize,
    _phantom: PhantomData<T>,
}

impl<T: 'static + TpCondSwap + Clone> SmallLSOram<T> {
    pub fn new(n: usize) -> Self {
        assert!(n > 0);
        assert!(size_of::<T>() <= 64);
        let elems_per_block = 64 / size_of::<T>();
        let n_blocks = (n + elems_per_block - 1) / elems_per_block;
        let mut inner = align_first::<A64, A64>(n_blocks);
        inner.resize(n_blocks, A64::default());
        Self {
            inner,
            n,
            elems_per_block,
            _phantom: PhantomData,
        }
    }

    pub fn from_slice(src: &[T]) -> Self {
        let mut src_iter = src.iter();
        let mut new = Self::new(src.len());
        for block in new.inner.iter_mut() {
            for bytes in block
                .as_bytes_mut()
                .chunks_mut(size_of::<T>())
                .take(new.elems_per_block)
            {
                if let Some(elem) = src_iter.next() {
                    unsafe { *(bytes.as_mut_ptr() as *mut T) = elem.to_owned() };
                } else {
                    break;
                }
            }
        }

        let src_slice = unsafe {
            std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * size_of::<T>())
        };
        new.inner.as_bytes_mut()[..src_slice.len() * size_of::<T>()].copy_from_slice(src_slice);
        new
    }

    pub fn as_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.n);
        let mut n = self.n;
        for block in self.inner.iter() {
            for bytes in block
                .as_bytes()
                .chunks(size_of::<T>())
                .take(self.elems_per_block)
            {
                let elem = unsafe { (*(bytes.as_ptr() as *const T)).clone() };
                out.push(elem);
                n -= 1;
                if n == 0 {
                    break;
                }
            }
        }
        out
    }

    pub fn cond_obliv_read(&self, index: TpU32, cond: TpBool) -> T {
        let (block_index, index_in_block) = self.index_to_block(index);
        let start_byte = index_in_block * size_of::<T>() as u32;
        let end_byte = start_byte + size_of::<T>() as u32;
        let start_byte = start_byte.expose() as usize;
        let end_byte = end_byte.expose() as usize;
        let mut out: Option<T> = None;
        for (i, block) in self.inner.iter().enumerate() {
            let mut val = unsafe {
                (*((&block.as_bytes()[start_byte..end_byte]).as_ptr() as *const T)).clone()
            };
            if let Some(out) = out.as_mut() {
                let is_target_block = block_index.tp_eq(&(i as u32)) & cond;
                is_target_block.cond_swap(&mut val, out);
            } else {
                out = Some(val);
            }
        }
        out.unwrap()
    }

    pub fn obliv_read(&self, index: TpU32) -> T {
        self.cond_obliv_read(index, TpBool::protect(true))
    }

    pub fn cond_obliv_write(&mut self, mut src: T, index: TpU32, cond: TpBool) {
        let (block_index, index_in_block) = self.index_to_block(index);
        let start_byte = index_in_block * size_of::<T>() as u32;
        let end_byte = start_byte + size_of::<T>() as u32;
        let start_byte = start_byte.expose() as usize;
        let end_byte = end_byte.expose() as usize;
        for (i, block) in self.inner.iter_mut().enumerate() {
            let is_target_block = block_index.tp_eq(&(i as u32)) & cond;
            let mut updated_val = unsafe {
                (*((&block.as_bytes()[start_byte..end_byte]).as_ptr() as *const T)).clone()
            };
            is_target_block.cond_swap(&mut src, &mut updated_val);
            let updated_val_slice = unsafe {
                std::slice::from_raw_parts(&updated_val as *const _ as *const u8, size_of::<T>())
            };
            block.as_bytes_mut()[start_byte..end_byte].copy_from_slice(updated_val_slice);
        }
    }

    pub fn cond_obliv_modify_with(&mut self, index: TpU32, f: impl Fn(&mut T), cond: TpBool) {
        let (block_index, index_in_block) = self.index_to_block(index);
        let start_byte = index_in_block * size_of::<T>() as u32;
        let end_byte = start_byte + size_of::<T>() as u32;
        let start_byte = start_byte.expose() as usize;
        let end_byte = end_byte.expose() as usize;
        for (i, block) in self.inner.iter_mut().enumerate() {
            let is_target_block = block_index.tp_eq(&(i as u32)) & cond;
            let mut val = unsafe {
                (*((&block.as_bytes()[start_byte..end_byte]).as_ptr() as *const T)).clone()
            };
            let mut updated_val = val.clone();
            f(&mut updated_val);
            is_target_block.cond_swap(&mut val, &mut updated_val);
            let val_slice = unsafe {
                std::slice::from_raw_parts(&val as *const _ as *const u8, size_of::<T>())
            };
            block.as_bytes_mut()[start_byte..end_byte].copy_from_slice(val_slice);
        }
    }

    pub fn obliv_modify_with(&mut self, index: TpU32, f: impl Fn(&mut T)) {
        self.cond_obliv_modify_with(index, f, TpBool::protect(true));
    }

    pub fn obliv_write(&mut self, src: T, index: TpU32) {
        self.cond_obliv_write(src, index, TpBool::protect(true))
    }

    // return (block_index, index_in_block)
    fn index_to_block(&self, index: TpU32) -> (TpU32, TpU32) {
        crate::utils::tp_u32_div(
            index,
            TpU32::protect(self.elems_per_block as u32),
            (self.inner.len() * self.elems_per_block) as u32,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn small_ls_oram() {
        use timing_shield::TpU64;
        let val = TpU64::protect(123456);
        let index = TpU32::protect(188);
        let mut oram = SmallLSOram::<TpU64>::new(1000);
        oram.obliv_write(val, index);
        let res = oram.obliv_read(index);
        assert_eq!(val.expose(), res.expose());
        oram.obliv_modify_with(index, |v| *v = *v + 1);
        let res = oram.obliv_read(index);
        assert_eq!(val.expose() + 1, res.expose());
    }
}
