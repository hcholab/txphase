use crate::mcmc::index_map::PackedIndexMapSlice;
use crate::{Bool, Genotype, UInt, Usize, U16, U32};
use common::ref_panel::BlockSlice;
use ndarray::{s, Array1, Array2, ArrayView1, Zip};

use std::time::{Duration, Instant};

use std::cell::RefCell;
thread_local! {
    pub static FILTER_1: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static FILTER_2: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static FILTER_3: RefCell<Duration> = RefCell::new(Duration::ZERO);
    pub static FILTER_4: RefCell<Duration> = RefCell::new(Duration::ZERO);
}

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
    //let mut s = 0;
    //let group_size = 5;
    //let mut new_neighbors: Vec<U32> = Vec::new();
    //let mut count = 0;
    //let mut tmp = Some(std::collections::HashMap::<u32, usize>::new());
    //for n in neighbors {
    //if n.is_none() {
    //continue;
    //}
    //let n = n.as_ref().unwrap();
    //s = n.len();
    //if count > group_size {
    //count = 0;
    //let mut vote = tmp.take().unwrap().into_iter().collect::<Vec<_>>();
    //vote.sort_by(|a, b| b.1.cmp(&a.1));
    //new_neighbors.extend(vote[..s].iter().map(|v| U32::protect(v.0)));
    //tmp = Some(std::collections::HashMap::new());
    //}
    //count += 1;
    //for i in n {
    //let i = i.expose();
    //tmp.as_mut()
    //.unwrap()
    //.entry(i)
    //.and_modify(|e| *e += 1)
    //.or_insert(1);
    //}
    //}
    //let mut vote = tmp.take().unwrap().into_iter().collect::<Vec<_>>();
    //vote.sort_by_key(|v| v.1);
    //new_neighbors.extend(vote[..s].iter().map(|v| U32::protect(v.0)));
    //let mut neighbors = new_neighbors;

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

    //TODO remove this part
    neighbors.iter_mut().zip(filter.iter()).for_each(|(n, &b)| {
        *n |= (!b).as_u32() << 31;
    });
    obliv_utils::bitonic_sort::bitonic_sort(&mut neighbors, true);
    let mut filter = Vec::with_capacity(neighbors.len());
    let mask = !(1 << 31);
    for n in &mut neighbors {
        let b = (*n >> 31).tp_eq(&1);
        filter.push(!b);
        *n &= mask;
    }

    //neighbors.resize(n_full_states.expose() as usize, U32::protect(0));
    //filter.resize(n_full_states.expose() as usize, Bool::protect(false));

    (neighbors, filter, n_full_states)
}

pub fn filter_blocks<'a>(
    neighbors: &[Option<Vec<U32>>],
    blocks: &[BlockSlice<'a>],
    packed_index_map: &PackedIndexMapSlice<'a>,
) -> (Array2<Genotype>, Array1<Bool>, Usize) {
    let window_len = neighbors.len();

    let t = Instant::now();

    let (max_k_neighbors, filter, n_full_states) = neighbors_to_filter(neighbors);

    FILTER_1.with(|v| {
        let mut v = v.borrow_mut();
        *v += t.elapsed();
    });

    let mut unfolded = Array2::from_elem((window_len, max_k_neighbors.len()), Genotype::protect(0));

    let unique_neighbors = {
        let t = Instant::now();

        let unpacked = packed_index_map.filter_and_unpack(&max_k_neighbors);

        FILTER_2.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        {
            let neighbor_set = neighbors
                .iter()
                .filter_map(|v| v.as_ref())
                .flatten()
                .map(|v| v.expose() as usize)
                .collect::<std::collections::HashSet<_>>();
            let mut neighbor_set_ref = neighbor_set.into_iter().collect::<Vec<_>>();
            neighbor_set_ref.sort();
            let neighbor_set = max_k_neighbors
                .iter()
                .take(n_full_states.expose() as usize)
                .map(|v| v.expose() as usize)
                .collect::<Vec<_>>();
            assert_eq!(neighbor_set, neighbor_set_ref);

            let unpacked = unpacked
                .iter()
                .map(|v| v.into_iter().map(|v| v.expose()).collect::<Vec<_>>())
                .take(n_full_states.expose() as usize)
                .collect::<Vec<_>>();

            for (i, test) in unpacked.iter().enumerate() {
                let reference = blocks
                    .iter()
                    .map(|block| block.index_map[neighbor_set[i]])
                    .collect::<Vec<_>>();
                if test != &reference {
                    println!();
                    println!("test {i}:\t\t {:?}", test);
                    println!("reference {i}:\t {:?}", reference);
                    panic!();
                }
            }
            //assert_eq!(unpacked, unpacked_ref);
        }

        let t = Instant::now();

        let x = unpacked.into_iter().fold(
            vec![Vec::<U16>::with_capacity(max_k_neighbors.len()); blocks.len()],
            |mut accu, unpacked| {
                for (a, b) in unpacked.into_iter().zip(accu.iter_mut()) {
                    b.push(a);
                }
                accu
            },
        );
        FILTER_3.with(|v| {
            let mut v = v.borrow_mut();
            *v += t.elapsed();
        });

        x
    };

    let mut start_slice = 0;
    let t = Instant::now();
    for (block, unique_neighbors) in blocks.into_iter().zip(unique_neighbors.into_iter()) {
        unfold_block(
            block,
            &unique_neighbors,
            unfolded.slice_mut(s![start_slice..start_slice + block.n_sites(), ..]),
        );
        start_slice += block.n_sites();
    }
    FILTER_4.with(|v| {
        let mut v = v.borrow_mut();
        *v += t.elapsed();
    });

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
