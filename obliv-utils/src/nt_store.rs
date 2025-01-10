use core::arch::x86_64::_mm_sfence;
use core::simd::u64x8;
use std::mem::size_of;
use timing_shield::{TpBool, TpCondSwap, TpEq, TpU32, TpU8};
use tp_fixedpoint::TpU64x8;

// The number of cachelines in a page, keeping one empty for a dummy write.
const PAGE_CAP: usize = 63;

pub struct OblivNtStore(Vec<Page>);

impl OblivNtStore {
    pub fn obliv_filter<'a, T: Clone + 'a>(
        cachelines: impl Iterator<Item = &'a T>,
        filter: impl Iterator<Item = TpBool>,
        n_cache_lines: usize,
    ) -> (Self, TpU32) {
        assert_eq!(std::mem::size_of::<T>(), 64);
        assert_eq!(std::mem::align_of::<T>(), 64);
        let n_pages = n_cache_lines.div_ceil(PAGE_CAP);
        let mut target = (0..n_pages).map(|_| Page::new()).collect::<Vec<_>>();

        let mut cacheline_count = TpU8::protect(0);
        let mut page_count = TpU32::protect(0);

        for (max_cacheline_count, (b, src_cacheline)) in filter.zip(cachelines).enumerate() {
            let max_page_count = max_cacheline_count.div_ceil(PAGE_CAP);

            for (i, target_page) in target.iter_mut().take(max_page_count).enumerate() {
                let cond = page_count.tp_eq(&(i as u32)) & b;
                let cacheline_index = cond.select(cacheline_count, TpU8::protect(PAGE_CAP as u8));

                unsafe {
                    let src_cacheline = &*(src_cacheline as *const _ as *const Cacheline);
                    target_page.obliv_write(cacheline_index, src_cacheline);
                }
            }
            cacheline_count += b.as_u8();
            let cond = cacheline_count.tp_eq(&(PAGE_CAP as u8));
            cacheline_count = cond.select(TpU8::protect(0), cacheline_count);
            page_count += cond.as_u32();
        }

        unsafe { _mm_sfence() };

        (
            Self(target),
            page_count * PAGE_CAP as u32 + cacheline_count.as_u32(),
        )
    }

    pub fn obliv_map<'a, T: Clone + 'a>(
        cachelines: impl Iterator<Item = &'a T>,
        indices: impl Iterator<Item = TpU32>,
        n_max_cache_lines: usize,
    ) -> Self {
        assert_eq!(std::mem::size_of::<T>(), 64);
        assert_eq!(std::mem::align_of::<T>(), 64);

        let n_pages = n_max_cache_lines.div_ceil(PAGE_CAP);
        let mut target = (0..n_pages).map(|_| Page::new()).collect::<Vec<_>>();
        for (target_index, src_cacheline) in indices.zip(cachelines) {
            let (page_index, cacheline_index) =
                Self::cal_ind(target_index, n_max_cache_lines as u32);
            for (i, target_page) in target.iter_mut().enumerate() {
                let cond = page_index.tp_eq(&(i as u32));
                let cacheline_index = cond.select(cacheline_index, TpU8::protect(PAGE_CAP as u8));
                unsafe {
                    let src_cacheline = &*(src_cacheline as *const _ as *const Cacheline);
                    target_page.obliv_write(cacheline_index, src_cacheline);
                }
            }
        }
        Self(target)
    }

    pub fn iter<'a, T: 'a>(&'a self) -> impl Iterator<Item = &'a T> {
        assert_eq!(std::mem::size_of::<T>(), 64);
        self.0
            .iter()
            .map(|page| page.0.iter().take(PAGE_CAP))
            .flatten()
            .map(|cacheline| unsafe { &(*(cacheline as *const _ as *const T)) })
    }

    // quickly calculate i / 63 and i % 63
    fn cal_ind(mut i: TpU32, max: u32) -> (TpU32, TpU8) {
        let rounds = (32 - max.leading_zeros()).div_ceil(6);
        let mut q_result = TpU32::protect(0);
        for _ in 0..rounds {
            let q = i >> 6;
            let r = i & 0b111111;
            q_result += q;
            i = q + r;
        }
        let mut i = i.as_u8();
        let cond = i.tp_eq(&63);
        q_result += cond.as_u32();
        i = cond.select(TpU8::protect(0), i);
        (q_result, i)
    }
}

use core::arch::x86_64::{__m512i, _mm512_stream_si512, _mm_clflush};

#[derive(Clone, Copy)]
#[repr(C, align(4096))]
pub struct Page(pub [Cacheline; 64]);

#[allow(dead_code)]
impl Page {
    pub fn new() -> Self {
        let mut new_self = Self([Cacheline::default(); 64]);
        new_self.flush();
        new_self
    }

    pub const fn as_slice<T>(&self) -> &[T] {
        assert!(std::mem::size_of::<T>() <= std::mem::size_of::<u64x8>());
        unsafe {
            &*std::ptr::slice_from_raw_parts(self.0.as_ptr() as *const T, 4096 / size_of::<T>())
        }
    }

    pub const fn as_mut_slice<T>(&mut self) -> &mut [T] {
        assert!(std::mem::size_of::<T>() <= std::mem::size_of::<u64x8>());
        unsafe {
            &mut *std::ptr::slice_from_raw_parts_mut(
                self.0.as_mut_ptr() as *mut T,
                4096 / size_of::<T>(),
            )
        }
    }

    #[inline]
    pub unsafe fn obliv_write(&mut self, index: TpU8, cacheline: &Cacheline) {
        _mm512_stream_si512(
            self.0[index.as_u64().expose() as usize]
                .as_mut_slice::<i32>()
                .as_mut_ptr(),
            *cacheline.as_ref::<__m512i>(),
        );
    }

    fn flush(&mut self) {
        unsafe {
            for cacheline in &mut self.0 {
                _mm_clflush(&cacheline.0 as *const _ as *const u8);
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct Cacheline(pub u64x8);

impl Cacheline {
    pub const fn as_ref<T>(&self) -> &T {
        assert!(std::mem::size_of::<T>() <= std::mem::size_of::<u64x8>());
        unsafe { &*(&self.0 as *const _ as *const T) }
    }

    pub const fn as_mut<T>(&mut self) -> &mut T {
        assert!(std::mem::size_of::<T>() <= std::mem::size_of::<u64x8>());
        unsafe { &mut *(&mut self.0 as *mut _ as *mut T) }
    }

    pub const fn as_slice<T>(&self) -> &[T] {
        assert!(std::mem::size_of::<T>() <= std::mem::size_of::<u64x8>());
        unsafe {
            &*std::ptr::slice_from_raw_parts(
                &self.0 as *const _ as *const T,
                64 / std::mem::size_of::<T>(),
            )
        }
    }

    pub const fn as_mut_slice<T>(&mut self) -> &mut [T] {
        assert!(std::mem::size_of::<T>() <= std::mem::size_of::<u64x8>());
        unsafe {
            &mut *std::ptr::slice_from_raw_parts_mut(
                &mut self.0 as *mut _ as *mut T,
                64 / std::mem::size_of::<T>(),
            )
        }
    }

    #[inline]
    pub fn obliv_write<T: Clone>(&mut self, index: TpU8, item: &T) {
        self.as_mut_slice::<T>()[index.as_u64().expose() as usize] = item.clone();
    }

    #[inline]
    pub fn obliv_read<T: Clone>(&self, index: TpU8) -> T {
        self.as_slice::<T>()[index.as_u64().expose() as usize].clone()
    }
}

impl TpCondSwap for Cacheline {
    fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
        TpU64x8::tp_cond_swap(cond, a.as_mut::<TpU64x8>(), b.as_mut::<TpU64x8>());
    }
}
