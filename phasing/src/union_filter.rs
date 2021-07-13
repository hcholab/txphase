use oram_sgx::align::A64Bytes;
use oram_sgx::utils::tp_u32_div;
use oram_sgx::*;
use rand::Rng;
use std::time::Instant;
use timing_shield::{TpBool, TpU32};

struct OblivBitmap<C: WriteOnlyORAMBackendCreator<64>>
where
    <C as BaseORAMBackendCreator<64>>::ORAM: WriteOnlyORAMBackend<64>,
{
    oram: SmallORAM<C>,
    n_bits: usize,
    n_bytes: usize,
    //s: usize,
    //s_mask: u8,
}

impl<C: WriteOnlyORAMBackendCreator<64>> OblivBitmap<C>
where
    <C as BaseORAMBackendCreator<64>>::ORAM: WriteOnlyORAMBackend<64>,
{
    pub fn new(n_bits: usize, s: usize, oram_backend_creator: C) -> Self {
        assert!(s <= 8);
        let n_bytes = (n_bits + 7) / 8;
        //let s_mask = (1u8 << s) - 1;
        Self {
            oram: SmallORAM::new(n_bytes as u32, 1, oram_backend_creator),
            n_bits,
            n_bytes,
            //s,
            //s_mask,
        }
    }

    pub fn set(&mut self, index: TpU32) {
        let (byte_index, bit_index) = tp_u32_div(index, TpU32::protect(8), self.n_bits as u32);
        let set_byte = 1 << bit_index.expose();
        //println!("second_byte = {:0b}", set_byte);
        //let second_byte = ((self.s_mask as u16) >> (8 - bit_index)) as u8;
        self.oram.modify_block_with(byte_index, |byte| {
            //println!("set_byte = \t {:08b}", set_byte);
            //println!("byte before = \t {:08b}", byte[0]);
            byte[0] |= set_byte;
            //println!("byte after = \t {:08b}", byte[0]);
        });
        //self.oram
        //.modify_block_with(byte_index + 1, |byte| byte[0] |= second_byte);
    }

    //pub fn set(&mut self, start_index: usize) {
    //let byte_index = start_index / 8;
    //let bit_index = start_index % 8;
    //let first_byte = self.s_mask << bit_index;
    //let second_byte = ((self.s_mask as u16) >> (8 - bit_index)) as u8;
    //self.oram
    //.modify_block_with(byte_index, |byte| byte[0] |= first_byte);
    //self.oram
    //.modify_block_with(byte_index + 1, |byte| byte[0] |= second_byte);
    //}

    pub fn into_iter(self) -> Box<dyn Iterator<Item = bool>> {
        let mut remaining_bits = self.n_bits;
        Box::new(
            self.oram
                .into_iter()
                .take(self.n_bytes)
                .map(move |byte| {
                    let byte = byte[0];
                    let take_n_bits = remaining_bits.min(8);
                    remaining_bits -= take_n_bits;
                    (0..take_n_bits)
                        .map(move |i| ((byte >> i) & 1) != 0)
                        .collect::<Vec<_>>()
                })
                .flatten(),
        )
    }
}

fn gen_sets(n_samples: usize, n_markers: usize, s: usize) -> Vec<Vec<usize>> {
    let mut rng = rand::thread_rng();
    (0..n_markers)
        .map(|_| (0..s).map(|_| rng.gen_range(0..n_samples)).collect())
        .collect()
}

const N_BYTES: usize = 4096;
const N_MARKERS: usize = N_BYTES * 8; 

fn gen_haplotypes(n_samples: usize) -> Vec<A64Bytes<N_BYTES>> {
    use rand::RngCore;
    let mut rng = rand::thread_rng();
    (0..n_samples)
        .map(|_| {
            let mut haplotype = A64Bytes::<N_BYTES>::default();
            rng.fill_bytes(haplotype.as_mut_slice());
            haplotype
        })
        .collect()
}

pub fn union_filter() {
    let n_samples = 100000;
    let capacity = 10000;
    let s = 4;

    let n_markers = N_MARKERS ;
    let sets = gen_sets(n_samples, n_markers, s);

    let now = Instant::now();
    let oram_creator = LinearScanningORAMCreator;
    let mut bitmap = OblivBitmap::new(n_samples, s, oram_creator);
    let init_ms = (Instant::now() - now).as_millis();
    println!(
        "Initialize oblivious bitmap of size {} samples: {} ms",
        n_samples, init_ms
    );

    let now = Instant::now();
    for i in sets.into_iter().flatten() {
        bitmap.set(TpU32::protect(i as u32));
    }
    let set_ms = (Instant::now() - now).as_millis();
    println!(
        "Set oblivious bitmap for {} markers: {} ms",
        n_markers, set_ms
    );

    let haplotypes = gen_haplotypes(n_samples);

    let bitmap = bitmap
        .into_iter()
        .map(|b| TpBool::protect(b))
        .collect::<Vec<_>>();
    let now = Instant::now();
    let (_selected_haplotypes, len) = obliv_filter(&bitmap, &haplotypes, capacity);
    let filter_ms = (Instant::now() - now).as_millis();
    println!("Filter haplotypes with filter capacity of {} haplotypes: {} ms", capacity, filter_ms);
    println!("# of selected haplotyps: {}", len.expose());
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn obliv_bitmap() {
        use std::collections::HashSet;
        let n_bits = 100;
        let n_indices = 10;
        let s = 4;
        let mut bitmap = OblivBitmap::new(n_bits, s, LinearScanningORAMCreator);
        //let mut bitmap = OblivBitmap::new(n_bits, s, LeakyORAMCreator);
        let mut rng = rand::thread_rng();
        let random_indices = (0..n_indices)
            .map(|_| rng.gen_range(0..n_bits))
            .collect::<HashSet<_>>();

        for &i in &random_indices {
            bitmap.set(TpU32::protect(i as u32));
        }
        let false_indices = (0..n_bits)
            .filter(|i| !random_indices.contains(i))
            .collect::<Vec<_>>();

        let results = bitmap.into_iter().collect::<Vec<_>>();
        for i in random_indices {
            assert!(results[i]);
        }

        for i in false_indices {
            assert!(!results[i]);
        }
    }
}
