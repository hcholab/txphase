use common::ref_panel::RefPanel;
use std::simd::{u64x8, Mask};
use timing_shield::{TpU16, TpU32};
use tp_fixedpoint::TpU64x8;

pub struct PackedIndexMap {
    packed: Vec<Vec<u64x8>>,
    block_map: Vec<usize>,
    slice_map: Vec<(usize, u8, u8)>,
    end_slice_map: Vec<usize>,
    bit_counts: Vec<u8>,
}

impl PackedIndexMap {
    pub fn new(ref_panel: &RefPanel) -> Self {
        let n_full_haps = ref_panel.n_haps;
        let mut packed = Vec::new();
        let mut cur_bit_count = 0;
        let mut cur_simd_count = 0;
        let mut cur_packed = Some(vec![u64x8::default(); n_full_haps]);
        let mut slice_map = Vec::with_capacity(ref_panel.blocks.len());
        let mut end_slice_map = Vec::with_capacity(ref_panel.blocks.len());
        let mut bit_counts = Vec::with_capacity(ref_panel.blocks.len());

        for block in &ref_panel.blocks {
            let n_bits = block.n_unique().next_power_of_two().ilog2();
            slice_map.push((packed.len(), cur_simd_count as u8, cur_bit_count as u8));
            bit_counts.push(n_bits as u8);

            for (&i, p) in block
                .index_map
                .iter()
                .zip(cur_packed.as_mut().unwrap().iter_mut())
            {
                p.as_mut_array()[cur_simd_count] |= (i as u64) << cur_bit_count;
            }
            cur_bit_count += n_bits;
            if cur_bit_count >= 64 {
                cur_simd_count += 1;
                cur_bit_count &= 0b111111;
                if cur_simd_count == 8 {
                    cur_simd_count = 0;
                    packed.push(cur_packed.take().unwrap());
                    cur_packed = Some(vec![u64x8::default(); n_full_haps]);
                }

                if cur_bit_count > 0 {
                    let n_shifts = n_bits - cur_bit_count;
                    for (&i, p) in block
                        .index_map
                        .iter()
                        .zip(cur_packed.as_mut().unwrap().iter_mut())
                    {
                        p.as_mut_array()[cur_simd_count] |= (i as u64) >> n_shifts;
                    }
                }
            }
            end_slice_map.push(packed.len());
        }
        if cur_bit_count > 0 || cur_simd_count > 0 {
            packed.push(cur_packed.take().unwrap());
        }
        Self {
            packed,
            slice_map,
            end_slice_map,
            bit_counts,
            block_map: ref_panel.block_map.clone(),
        }
    }

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> PackedIndexMapSlice<'a> {
        let start_block_id = match self.block_map.binary_search(&start) {
            Ok(i) => i + 1,
            Err(i) => i,
        };
        let end_block_id = match self.block_map.binary_search(&(end)) {
            Ok(i) => i,
            Err(i) => i,
        };

        let (start_packed, start_simd_count, start_bit_count) = self.slice_map[start_block_id];
        let end_packed = self.end_slice_map[end_block_id];

        let packed = if end_packed + 1 >= self.packed.len() {
            &self.packed[start_packed..]
        } else {
            &self.packed[start_packed..end_packed + 1]
        };

        let bit_counts = if end_block_id + 1 >= self.bit_counts.len() {
            &self.bit_counts[start_block_id..]
        } else {
            &self.bit_counts[start_block_id..end_block_id + 1]
        };

        let total_bits = bit_counts.iter().fold(0usize, |acc, e| acc + *e as usize);

        let packed = if total_bits.div_ceil(512) < packed.len() {
            Packed::Wrapped((
                Self::wrap(
                    &packed[0],
                    packed.last().unwrap(),
                    start_simd_count as usize,
                    start_bit_count as usize,
                ),
                &packed[1..packed.len() - 1],
            ))
        } else {
            Packed::Unwrapped(packed)
        };

        PackedIndexMapSlice {
            packed,
            bit_counts,
            start: (start_simd_count, start_bit_count),
        }
    }

    fn wrap(
        head: &[u64x8],
        tail: &[u64x8],
        start_simd_count: usize,
        start_bit_count: usize,
    ) -> Vec<u64x8> {
        let simd_mask = !((1u64 << (start_simd_count + 1)) - 1);
        let simd_mask = Mask::from_bitmask(simd_mask);
        let bit_mask = (1 << start_bit_count) - 1;

        head.into_iter()
            .zip(tail.into_iter())
            .map(|(h, t)| {
                let mut wrapped =
                    unsafe { u64x8::load_select_unchecked(h.as_array().as_slice(), simd_mask, *t) };
                wrapped.as_mut_array()[start_simd_count] = (h.as_array()[start_simd_count]
                    & !bit_mask)
                    | (t.as_array()[start_simd_count] & bit_mask);
                wrapped
            })
            .collect()
    }
}

pub struct PackedIndexMapSlice<'a> {
    pub packed: Packed<'a>,
    bit_counts: &'a [u8],
    start: (u8, u8),
}

impl<'a> PackedIndexMapSlice<'a> {
    pub fn filter_and_unpack(&self, indices: &[TpU32]) -> Vec<Vec<TpU16>> {
        //let filtered_packed = self.filter_debug(indices);
        let filtered_packed = self.filter_linear_scanning(indices);
        self.unpack(filtered_packed)
    }

    pub fn filter_linear_scanning(&self, indices: &[TpU32]) -> Vec<Vec<TpU64x8>> {
        let mut bitmap = obliv_utils::bitmap::OblivBitmap::new(self.packed.n_rows());
        for &i in indices {
            bitmap.set(i);
        }

        let mut filtered_packed = vec![Vec::<TpU64x8>::new(); indices.len()];
        match &self.packed {
            Packed::Wrapped((w, p)) => {
                {
                    let (result, _) = obliv_utils::nt_store::OblivNtStore::obliv_filter(
                        w.iter(),
                        bitmap.iter(),
                        indices.len(),
                    );
                    for (r, f) in result
                        .iter::<TpU64x8>()
                        .cloned()
                        .zip(filtered_packed.iter_mut())
                    {
                        f.push(r);
                    }
                }
                for packed in p.iter() {
                    let (result, _) = obliv_utils::nt_store::OblivNtStore::obliv_filter(
                        packed.iter(),
                        bitmap.iter(),
                        indices.len(),
                    );
                    for (r, f) in result
                        .iter::<TpU64x8>()
                        .cloned()
                        .zip(filtered_packed.iter_mut())
                    {
                        f.push(r);
                    }
                }
            }
            Packed::Unwrapped(p) => {
                for packed in p.iter() {
                    let (result, _) = obliv_utils::nt_store::OblivNtStore::obliv_filter(
                        packed.iter(),
                        bitmap.iter(),
                        indices.len(),
                    );
                    for (r, f) in result
                        .iter::<TpU64x8>()
                        .cloned()
                        .zip(filtered_packed.iter_mut())
                    {
                        f.push(r);
                    }
                }
            }
        };

        filtered_packed
    }

    pub fn filter_debug(&self, indices: &[TpU32]) -> Vec<Vec<TpU64x8>> {
        let indices = indices
            .iter()
            .map(|v| v.expose() as usize)
            .collect::<Vec<_>>();
        let mut filtered_packed = vec![Vec::<TpU64x8>::new(); indices.len()];
        match &self.packed {
            Packed::Wrapped((w, p)) => {
                for (j, &i) in indices.iter().enumerate() {
                    filtered_packed[j].push(TpU64x8::protect(w[i].clone()));
                }
                for packed in p.iter() {
                    for (j, &i) in indices.iter().enumerate() {
                        filtered_packed[j].push(TpU64x8::protect(packed[i].clone()));
                    }
                }
            }
            Packed::Unwrapped(p) => {
                for packed in p.iter() {
                    for (j, &i) in indices.iter().enumerate() {
                        filtered_packed[j].push(TpU64x8::protect(packed[i].clone()));
                    }
                }
            }
        };
        filtered_packed
    }

    pub fn unpack(&self, filtered_packed: Vec<Vec<TpU64x8>>) -> Vec<Vec<TpU16>> {
        let mut unpacked = Vec::with_capacity(filtered_packed.len());
        for packed in filtered_packed {
            let packed = if let Packed::Wrapped(_) = self.packed {
                Self::unwrap(packed)
            } else {
                packed
            };

            let mut packed_iter = packed.into_iter();
            let mut cur_packed = packed_iter.next().unwrap();

            let mut cur_simd_count = self.start.0 as usize;
            let mut cur_bit_count = self.start.1 as u32;

            let mut cur_unpacked = Vec::with_capacity(self.bit_counts.len());
            for &n_bits in self.bit_counts {
                let n_bits = n_bits as u32;
                let mask = (1 << n_bits) - 1;
                let mut i =
                    ((cur_packed.as_array()[cur_simd_count] >> cur_bit_count) & mask).as_u16();

                cur_bit_count += n_bits;
                if cur_bit_count >= 64 {
                    cur_simd_count += 1;
                    cur_bit_count &= 0b111111;
                    if cur_simd_count == 8 {
                        cur_simd_count = 0;
                        if let Some(cur_packed_) = packed_iter.next() {
                            cur_packed = cur_packed_;
                        }
                    }

                    if cur_bit_count > 0 {
                        let n_shifts = n_bits - cur_bit_count;
                        let mask = (1 << cur_bit_count) - 1;
                        i |= ((cur_packed.as_array()[cur_simd_count] & mask) << n_shifts).as_u16();
                    }
                }
                cur_unpacked.push(i)
            }
            unpacked.push(cur_unpacked)
        }
        unpacked
    }

    fn unwrap(mut packed: Vec<TpU64x8>) -> Vec<TpU64x8> {
        packed.push(packed[0].clone());
        packed
    }
}

pub enum Packed<'a> {
    Wrapped((Vec<u64x8>, &'a [Vec<u64x8>])),
    Unwrapped(&'a [Vec<u64x8>]),
}

impl<'a> Packed<'a> {
    pub fn n_rows(&self) -> usize {
        match self {
            Self::Wrapped((w, _)) => w.len(),
            Self::Unwrapped(p) => p[0].len(),
        }
    }
}
