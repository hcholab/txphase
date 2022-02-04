// modified from https://github.com/mobilecoinofficial/mc-oblivious/blob/master/aligned-cmov/src/cmov_impl_asm.rs
use std::arch::asm;
use timing_shield::{TpBool, TpCondSwap};

#[repr(align(64))]
#[derive(Clone, Debug)]
pub struct A64Bytes<const N: usize>([u8; N]);

impl<const N: usize> A64Bytes<N> {
    pub fn from_slice(slice: &[u8]) -> Self {
        assert_eq!(N, slice.len());
        let mut new_self: Self = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        new_self.0.copy_from_slice(slice);
        new_self
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.0
    }

    // CMov for blocks aligned to 64-byte boundary
    #[inline]
    pub fn cmov(&mut self, condition: TpBool, src: &Self) {
        if N != 0 {
            // Without avx2, fallback to cmov_byte_slice_a8
            #[cfg(not(target_feature = "avx2"))]
            {
                let count = (N + 7) / 8;
                unsafe {
                    cmov_byte_slice_a8(
                        condition.expose(),
                        src.0.as_ptr() as *const u64,
                        self.0.as_mut_ptr() as *mut u64,
                        count,
                    )
                };
            }

            // If we have avx2 then we can use cmov_byte_slice_a64 implementation
            #[cfg(target_feature = "avx2")]
            {
                let count = (N + 63) / 64;
                unsafe {
                    // Note! API for count is different from with cmov_byte_slice_a8,
                    // because in the `[BASE + INDEX * SCALE + DISPLACEMENT]` syntax,
                    // scale > 8 not allowed in x86-64, here the scale would need to be 64.
                    // In cmov_byte_slice_a64, the 4th parameter is the number of bytes to copy.
                    cmov_byte_slice_a64(
                        condition.expose(),
                        src.0.as_ptr() as *const u64,
                        self.0.as_mut_ptr() as *mut u64,
                        count * 64,
                    )
                };
            }
        }
    }

    pub fn as_aligned_blocks(&self) -> &[A64Bytes<64>] {
        unsafe {
            let ptr = self.0.as_ptr() as *const A64Bytes<64>;
            assert!(ptr.align_offset(64) == 0);
            std::slice::from_raw_parts(ptr, N / 64)
        }
    }

    pub fn as_mut_aligned_blocks(&mut self) -> &mut [A64Bytes<64>] {
        unsafe {
            let ptr = self.0.as_mut_ptr() as *mut A64Bytes<64>;
            assert!(ptr.align_offset(64) == 0);
            std::slice::from_raw_parts_mut(ptr, N / 64)
        }
    }
}

impl<const N: usize> Default for A64Bytes<N> {
    fn default() -> Self {
        Self([0u8; N])
    }
}

impl<const N: usize> TpCondSwap for A64Bytes<N> {
    #[inline]
    fn tp_cond_swap(condition: TpBool, a: &mut Self, b: &mut Self) {
        let mut temp = Self::default();
        temp.cmov(condition, a);
        a.cmov(condition, b);
        b.cmov(condition, &temp);
    }
}

// Should be a constant time function equivalent to:
// if condition { memcpy(dest, src, count * 8) }
// for pointers aligned to 8 byte boundary. Assumes count > 0
#[inline]
#[allow(unused)]
pub unsafe fn cmov_byte_slice_a8(
    condition: bool,
    src: *const u64,
    dest: *mut u64,
    mut count: usize,
) {
    // Create a temporary to be assigned to a register for cmov
    // We also use it to pass the condition in, which is only needed in a register
    // before entering the loop.
    let mut temp: u64 = condition as u64;

    // The idea here is to test once before we enter the loop and reuse the test result
    // for every cmov.
    // The loop is a dec, jnz loop.
    // Because the semantics of x86 loops are, decrement, then test for 0, and
    // not test for 0 and then decrement, the values of the loop variable of 1..count
    // and not 0..count - 1. We adjust for this by subtracting 8 when indexing.
    // Because dec will clobber ZF, we can't use cmovnz. We store
    // the result of outer test in CF which is not clobbered by dec.
    // cmovc is contingent on CF
    // negb is used to set CF to 1 iff the condition value was 1.
    // Pity, we cannot cmov directly to memory
    //
    // temp is a "register output" because this is the way to obtain a scratch register
    // when we don't care what actual register is used and we want the compiler to pick.
    //
    // count is modified by the assembly -- for this reason it has to be labelled
    // as an input and an output, otherwise its illegal to modify it.
    //
    // The condition is passed in through temp, then neg moves the value to CF.
    // After that we don't need condition in a register, so temp register can be reused.
    asm!("neg {0}
            2:
            mov {0}, [{3} + 8*{1} - 8]
            cmovc {0}, [{2} + 8*{1} - 8]
            mov [{3} + 8*{1} - 8], {0} 
            dec {1} 
            jnz 2" ,
            inout(reg) temp,
            inout(reg) count,
            in(reg) src,
            in(reg) dest,
            options(nostack),
    );

    //llvm_asm!("neg $0
               //loop_body_${:uid}:
                 //mov $0, [$3 + 8*$1 - 8]
                 //cmovc $0, [$2 + 8*$1 - 8]
                 //mov [$3 + 8*$1 - 8], $0
                 //dec $1
                 //jnz loop_body_${:uid}"
            //: "+&r"(temp), "+&r"(count)
            //: "r"(src), "r"(dest)
            //: "cc", "memory"
            //: "volatile", "intel");
    // cc is because we are setting flags in test
    // memory is because we are dereferencing a bunch of pointers in asm
    // volatile is because the output variable "temp" is not the true output
    // of this block, the memory side-effects are.
}

// Should be a constant time function equivalent to:
// if condition { memcpy(dest, src, num_bytes) }
// for pointers aligned to *64* byte boundary. Will fault if that is not the case.
// Assumes num_bytes > 0, and num_bytes divisible by 64!
// This version uses AVX2 256-bit moves
#[cfg(target_feature = "avx2")]
#[inline]
#[allow(unused)]
pub unsafe fn cmov_byte_slice_a64(
    condition: bool,
    src: *const u64,
    dest: *mut u64,
    num_bytes: usize,
) {
    debug_assert!(
        num_bytes > 0,
        "num_bytes cannot be 0, caller must handle that"
    );
    debug_assert!(
        num_bytes % 64 == 0,
        "num_bytes must be divisible by 64, caller must handle that"
    );
    // Similarly as before, we want to test once and use the test result for the whole loop.
    //
    // Before we enter the loop, we want to set ymm1 to all 0s or all 1s, depending on condition.
    // We use neg to make it all 0s or all 1s 64bit, then vmovq to move that to xmm2, then vbroadcastsd
    // to fill ymm1 with 1s or zeros.
    //
    // This time the cmov mechanism is:
    // - VMOVDQA to load the source into a ymm register,
    // - VPMASKMOVQ to move that to memory, after masking it with ymm1.
    // An alternative approach which I didn't test is, MASKMOV to another ymm register,
    // then move that register to memory.
    //
    // Notes:
    // temp = $0 is the scratch register
    // We aren't allowed to modify condition or num_bytes = $3 unless
    // we annotate them appropriately as outputs.
    // So, num_bytes is an in/out register, and we pass condition in via
    // the scratch register, instead of a dedicated register.
    //
    // Once we have the mask in ymm1 we don't need the condition in a register anymore.
    // Then, $0 is the loop variable, counting down from num_bytes in steps of 64
    //
    // The values of $0 in the loop are num_bytes, num_bytes - 64, ... 64,
    // rather than num_bytes - 64 ... 0. This is because the semantics of the loop
    // are subtract 64, then test for 0, rather than test for 0, then subtract.
    // So when we index using $0, we subtract 64 to compensate.
    //
    // We unroll the loop once because we can assume 64 byte alignment, but ymm
    // register holds 32 bytes. So one round of vmovdqa, vpmaskmovq moves 32 bytes.
    // So we do this twice in one pass through the loop.
    //
    // TODO: In AVX512 zmm registers we could move 64 bytes at once...
    // TODO: Does unrolling the loop more help?
    let mut temp: u64 = condition as u64;

    asm!("neg {0}
            vmovq xmm2, {0}
            vbroadcastsd ymm1, xmm2
            mov {0}, {3}
            2:
            vmovdqa ymm2, [{1} + {0} - 64]
            vpmaskmovq [{2} + {0} - 64], ymm1, ymm2
            vmovdqa ymm3, [{1} + {0} - 32]
            vpmaskmovq [{2} + {0} - 32], ymm1, ymm3
            sub {0}, 64
            jnz 2 
        ",
        inout(reg) temp,
        in(reg) src,
        in(reg) dest,
        in(reg) num_bytes,
        options(nostack)
    );

    //llvm_asm!("neg $0
               //vmovq xmm2, $0
               //vbroadcastsd ymm1, xmm2
               //mov $0, $3
               //loop_body2_${:uid}:
                 //vmovdqa ymm2, [$1 + $0 - 64]
                 //vpmaskmovq [$2 + $0 - 64], ymm1, ymm2
                 //vmovdqa ymm3, [$1 + $0 - 32]
                 //vpmaskmovq [$2 + $0 - 32], ymm1, ymm3
                 //sub $0, 64
                 //jnz loop_body2_${:uid}"
            //: "+&r"(temp)
            //: "r"(src), "r"(dest), "rmi"(num_bytes)
            //: "cc", "memory", "ymm1", "ymm2", "ymm3"
            //: "volatile", "intel");
    // cc is because we are setting flags
    // memory is because we are dereferencing a bunch of pointers in asm
    // volatile is because the output variable "temp" is not the true output
    // of this block, the memory side-effects are.
}
