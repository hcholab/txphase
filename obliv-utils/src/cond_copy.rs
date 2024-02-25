use timing_shield::{TpBool, TpCondSwap};

pub fn cond_copy_slice_1<T: TpCondSwap + Clone>(tar: &mut [T], src: &[T], cond: TpBool) {
    assert_eq!(tar.len(), src.len());
    for (t, s) in tar.iter_mut().zip(src.iter()) {
        *t = cond.select(s.clone(), t.clone());
    }
}

pub fn cond_copy_slice_2<T: TpCondSwap + Clone>(tar: &mut [T], src: &[T], cond: TpBool) {
    assert_eq!(tar.len(), src.len());

    #[cfg(target_feature = "avx2")]
    {
        use crate::aligned::Aligned;
        let (src_prefix, src_aligned, src_suffix) = unsafe { src.align_to::<Aligned<u64>>() };
        let (tar_prefix, tar_aligned, tar_suffix) = unsafe { tar.align_to_mut::<Aligned<u64>>() };

        for (t, s) in tar_prefix.iter_mut().zip(src_prefix.iter()) {
            *t = cond.select(s.clone(), t.clone());
        }

        let num_bytes = tar_aligned.len() * std::mem::size_of::<Aligned<u64>>();

        use crate::cmov::cmov_byte_slice_a64;
        unsafe {
            cmov_byte_slice_a64(
                cond.expose(),
                src_aligned.as_ptr() as *const u64,
                tar_aligned.as_mut_ptr() as *mut u64,
                num_bytes as usize,
            );
        }

        for (t, s) in tar_suffix.iter_mut().zip(src_suffix.iter()) {
            *t = cond.select(s.clone(), t.clone());
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        let (src_prefix, src_aligned, src_suffix) = unsafe { src.align_to::<u64>() };
        let (tar_prefix, tar_aligned, tar_suffix) = unsafe { tar.align_to_mut::<u64>() };

        for (t, s) in tar_prefix.iter_mut().zip(src_prefix.iter()) {
            *t = cond.select(s.clone(), t.clone());
        }

        let count = tar_aligned.len() * std::mem::size_of::<T>() / 8;

        use crate::cmov::cmov_byte_slice_a8;
        unsafe {
            cmov_byte_slice_a8(
                cond.expose(),
                src_aligned.as_ptr() as *const u64,
                tar_aligned.as_mut_ptr() as *mut u64,
                count as usize,
            );
        }

        for (t, s) in tar_suffix.iter_mut().zip(src_suffix.iter()) {
            *t = cond.select(s.clone(), t.clone());
        }
    }
}

pub fn cond_copy_slice_3<T>(tar: &mut [T], src: &[T], cond: TpBool) {
    assert_eq!(tar.len(), src.len());
    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<u64>());
    use crate::cmov::cmov_u64;

    let cond = cond.expose();
    for (t, s) in tar.iter_mut().zip(src.iter()) {
        unsafe {
            cmov_u64(cond, std::mem::transmute(s), std::mem::transmute(t));
        }
    }
}

pub fn cond_copy_slice_4<T: Clone>(tar: &mut [T], src: &[T], cond: TpBool) {
    assert_eq!(tar.len(), src.len());
    if cond.expose() {
        tar.clone_from_slice(src);
    }
}
