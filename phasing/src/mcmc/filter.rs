use crate::{Bool, Genotype, UInt, Usize, U16, U32};
use common::ref_panel::BlockSlice;
use ndarray::{s, Array1, Array2, ArrayView1, Zip};
use obliv_utils::vec::OblivVec;
use tp_fixedpoint::timing_shield::TpU64;

#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::TpEq;

pub fn find_nn_bitmap(neighbors: &[Option<Vec<U32>>], n_haps: usize) -> (Vec<Bool>, UInt) {
    #[cfg(feature = "obliv")]
    let neighbors_bitmap = {
        let mut bitmap = obliv_utils::bitmap::OblivBitmap::new(n_haps);
        bitmap.map_from_iter(
            neighbors
                .into_iter()
                .filter_map(|v| v.as_ref())
                .map(|v| v.iter().cloned())
                .flatten(),
        );
        bitmap
    };

    #[cfg(not(feature = "obliv"))]
    let neighbors_bitmap = {
        let mut bitmap = vec![false; n_haps];
        for &i in neighbors
            .into_iter()
            .filter_map(|v| v.as_ref())
            .map(|v| v.iter())
            .flatten()
        {
            bitmap[i] = true;
        }
        bitmap
    };

    #[cfg(feature = "obliv")]
    let k = neighbors_bitmap
        .iter()
        .fold(UInt::protect(0), |acc, v| acc + v.as_u32());

    #[cfg(not(feature = "obliv"))]
    let k = neighbors_bitmap.iter().filter(|&&b| b).count() as u32;

    #[cfg(feature = "obliv")]
    return (neighbors_bitmap.iter().collect(), k);

    #[cfg(not(feature = "obliv"))]
    (neighbors_bitmap, k)
}

#[cfg(feature = "obliv")]
pub fn neighbors_to_filter(neighbors: &[Option<Vec<U32>>]) -> (Vec<U32>, Vec<Bool>, Usize) {
    let mut neighbors = neighbors
        .into_iter()
        .filter_map(|v| v.as_ref())
        .flatten()
        .cloned()
        .collect::<Vec<U32>>();
    obliv_utils::bitonic_sort::bitonic_sort(&mut neighbors, true);

    let mut n_full_states = Usize::protect(1);

    let filter = {
        let mut filter = vec![Bool::protect(false); neighbors.len()];
        let mut prev = neighbors[0];
        filter[0] = Bool::protect(true);

        for (f, n) in filter.iter_mut().zip(neighbors.iter()).skip(1) {
            let cond = prev.tp_not_eq(n);
            *f = cond;
            prev = *n;
            n_full_states = cond.select(n_full_states + 1, n_full_states);
        }
        filter
    };

    ////TODO remove this part
    //neighbors.iter_mut().zip(filter.iter()).for_each(|(n, &b)| {
    //*n |= (!b).as_u64() << 63;
    //});
    //obliv_utils::bitonic_sort::bitonic_sort(&mut neighbors, true);
    //let mut filter = Vec::with_capacity(neighbors.len());
    //let mask = !(1 << 63);
    //for n in &mut neighbors {
    //let b = (*n >> 63).tp_eq(&1);
    //filter.push(!b);
    //*n &= mask;
    //}

    (neighbors, filter, n_full_states)
}

pub fn filter_blocks<'a>(
    neighbors: &[Option<Vec<U32>>],
    blocks: &[BlockSlice<'a>],
) -> (Array2<Genotype>, Array1<Bool>, Usize) {
    let window_len = neighbors.len();

    let (max_k_neighbors, filter, n_full_states) = neighbors_to_filter(neighbors);

    let mut unfolded = Array2::from_elem((window_len, max_k_neighbors.len()), Genotype::protect(0));

    let unique_neighbors = {
        let packed = pack_index_maps(blocks);
        let mut max_k_neighbors_indices = vec![Vec::new(); max_k_neighbors.len()];

        for packed in packed.into_iter() {
            for (&n, indices) in max_k_neighbors
                .iter()
                .zip(max_k_neighbors_indices.iter_mut())
            {
                indices.push(packed.get(n));
            }
        }
        let unpacked = unpack_indices(&max_k_neighbors_indices, blocks);
        unpacked.into_iter().fold(
            vec![Vec::<U16>::with_capacity(blocks.len()); max_k_neighbors.len()],
            |mut accu, unpacked| {
                for (a, b) in unpacked.into_iter().zip(accu.iter_mut()) {
                    b.push(a);
                }
                accu
            },
        )
    };

    let mut start_slice = 0;
    for (block, unique_neighbors) in blocks.into_iter().zip(unique_neighbors.into_iter()) {
        unfold_block(
            block,
            &unique_neighbors,
            unfolded.slice_mut(s![start_slice..start_slice + block.n_sites(), ..]),
        );
        start_slice += block.n_sites();
    }

    let filter = Array1::from_vec(filter);

    (unfolded, filter, n_full_states)
}

pub fn unfold_block<'a>(
    block: &BlockSlice<'a>,
    unique_neighbors: &[U16],
    mut unfolded: ndarray::ArrayViewMut2<Genotype>,
) {
    assert_eq!(unfolded.nrows(), block.n_sites());
    assert_eq!(unfolded.ncols(), unique_neighbors.len());

    #[cfg(not(feature = "obliv"))]
    let unique_neighbors = neighbors
        .iter()
        .map(|&v| block.index_map[v])
        .collect::<Vec<_>>();

    Zip::from(block.haplotypes.rows())
        .and(unfolded.rows_mut())
        .for_each(|h, u| {
            unfold_haps(h, block.n_unique(), &unique_neighbors[..], u);
        });
}

fn unfold_haps(
    haps: ArrayView1<u8>,
    n_unique_haps: usize,
    unique_neighbors: &[U16],
    mut unfolded: ndarray::ArrayViewMut1<Genotype>,
) {
    #[cfg(feature = "obliv")]
    {
        let mut inner = obliv_utils::vec::OblivVec::with_capacity(haps.len().div_ceil(8));
        let haps = haps.as_slice().unwrap();
        for chunk in haps.chunks(8) {
            let mut new_u64 = 0u64;
            for &i in chunk.iter().rev() {
                new_u64 <<= 8;
                new_u64 |= i as u64;
            }
            inner.push(crate::U64::protect(new_u64));
        }

        let bitmap = obliv_utils::bitmap::OblivBitmap::from_inner(inner, n_unique_haps);

        return unfolded
            .iter_mut()
            .zip(unique_neighbors.into_iter())
            .for_each(|(b, n)| {
                *b = bitmap.get(n.as_u32()).as_i8();
            });
    }

    #[cfg(not(feature = "obliv"))]
    {
        let mut bitmap = bitvec::prelude::BitSlice::<_, bitvec::prelude::Lsb0>::from_slice(
            haps.as_slice().unwrap(),
        );
        return unfolded
            .iter_mut()
            .zip(unique_neighbors.into_iter())
            .for_each(|(b, &n)| {
                *b = bitmap[n as usize] as i8;
            });
    }
}

//#[cfg(feature = "obliv")]
//fn unfold_block_2<'a>(
//block: &common::ref_panel::BlockSlice<'a>,
//neighbors: &[Usize],
//mut unfolded: ndarray::ArrayViewMut2<Genotype>,
//) {
//assert_eq!(unfolded.nrows(), block.n_sites());
//assert_eq!(unfolded.ncols(), neighbors.len());

//use std::iter::FromIterator;
//use tp_fixedpoint::timing_shield::TpU16;

//let obliv_index_map =
//obliv_utils::vec::OblivVec::from_iter(block.index_map.iter().map(|&v| TpU16::protect(v)));

//let unique_neighbors = neighbors
//.iter()
//.map(|&v| obliv_index_map.get(v.as_u32()))
//.collect::<Vec<_>>();

//let transposed_block = block.transpose();
//use tp_fixedpoint::timing_shield::TpU8;

//for (u, mut unfolded_col) in unique_neighbors.iter().zip(unfolded.columns_mut()) {
//let mut hap = Vec::with_capacity(0);
//Zip::indexed(transposed_block.haplotypes.rows()).for_each(|i, r| {
//if i == 0 {
//hap = r.into_iter().map(|&v| TpU8::protect(v)).collect::<Vec<_>>();
//} else {
//for (j, &h) in hap.iter_mut().zip(r.into_iter()) {
//*j = u.tp_eq(&(i as u16)).select(TpU8::protect(h), *j);
//}
//}
//});

//let mut i = 0;
//for mut h in hap {
//for _ in 0..8 {
//unfolded_col[i] = (h & 1).as_i8();
//h >>= 1;
//i += 1;
//if i == unfolded_col.len() {
//break;
//}
//}
//if i == unfolded_col.len() {
//break;
//}
//}
//}
//}

pub fn pack_index_maps<'a>(blocks: &[BlockSlice<'a>]) -> Vec<OblivVec<TpU64>> {
    let n_full_haps = blocks[0].index_map.len();
    let mut packed = Vec::new();
    let mut cur_bit_count = 0;
    let mut cur_packed = Some(vec![0u64; n_full_haps]);
    for block in blocks {
        let n_bits = block.n_unique().next_power_of_two().ilog2();
        for (&i, p) in block
            .index_map
            .iter()
            .zip(cur_packed.as_mut().unwrap().iter_mut())
        {
            *p |= (i as u64) << cur_bit_count;
        }
        cur_bit_count += n_bits;
        if cur_bit_count >= 64 {
            packed.push(OblivVec::from_iter(
                cur_packed
                    .take()
                    .unwrap()
                    .into_iter()
                    .map(|v| TpU64::protect(v)),
            ));
            cur_packed = Some(vec![0u64; n_full_haps]);
            cur_bit_count &= 0b111111;

            if cur_bit_count > 0 {
                let n_shifts = n_bits - cur_bit_count;
                for (&i, p) in block
                    .index_map
                    .iter()
                    .zip(cur_packed.as_mut().unwrap().iter_mut())
                {
                    *p |= (i as u64) >> n_shifts;
                }
            }
        }
    }
    if cur_bit_count > 0 {
        packed.push(OblivVec::from_iter(
            cur_packed
                .take()
                .unwrap()
                .into_iter()
                .map(|v| TpU64::protect(v)),
        ));
    }
    packed
}

pub fn unpack_indices<'a>(packed: &[Vec<TpU64>], blocks: &[BlockSlice<'a>]) -> Vec<Vec<U16>> {
    let mut unpacked = vec![Vec::with_capacity(blocks.len())];
    let lens = blocks
        .into_iter()
        .map(|b| b.n_unique().next_power_of_two().ilog2())
        .collect::<Vec<_>>();
    for packed in packed {
        let mut packed_iter = packed.iter();
        let mut cur_packed = packed_iter.next().unwrap().to_owned();
        let mut n_bits = 64;
        let new_unpacked = lens
            .iter()
            .map(|&l| {
                if n_bits >= l {
                    let unpacked = cur_packed.as_u16() & ((1 << l) - 1);
                    cur_packed >>= l as u32;
                    n_bits -= l;
                    unpacked
                } else {
                    let mut unpacked = cur_packed.as_u16();
                    cur_packed = packed_iter.next().unwrap().to_owned();
                    let n_remain = l - n_bits;
                    unpacked |= (cur_packed.as_u16() & ((1 << n_remain) - 1)) << n_bits;
                    cur_packed >>= n_remain;
                    n_bits = 64 - n_remain;
                    unpacked
                }
            })
            .collect();
        unpacked.push(new_unpacked);
    }

    unpacked
}
