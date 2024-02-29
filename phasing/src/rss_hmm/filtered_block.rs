use crate::{Bool, Real, UInt};
use bitvec::{order::Lsb0, slice::BitSlice};
use common::ref_panel::BlockSlice;
use ndarray::{Array1, ArrayView1, ArrayView2};

#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::TpEq;

#[derive(Clone)]
pub struct FilteredBlockSliceObliv<'a> {
    pub index_map: ArrayView1<'a, u16>,
    pub haplotypes: ArrayView2<'a, u8>,
    pub weights: Array1<Real>,
    pub inv_weights: Array1<Real>,
    pub n_unique_states: UInt,
    //pub filter: Array1<Bool>,
    pub full_filter: Array1<Bool>,
}

impl<'a> FilteredBlockSliceObliv<'a> {
    pub fn from_block_slice(block_slice: &BlockSlice<'a>, full_filter: &[Bool]) -> Self {
        #[cfg(feature = "obliv")]
        let mut filter = Array1::<Bool>::from_elem(block_slice.n_unique(), Bool::protect(false));

        #[cfg(not(feature = "obliv"))]
        let mut filter = Array1::<Bool>::from_elem(block_slice.n_unique(), false);

        #[cfg(feature = "obliv")]
        let mut weights = Array1::<UInt>::from_elem(block_slice.n_unique(), UInt::protect(0));

        #[cfg(not(feature = "obliv"))]
        let mut weights = Array1::<UInt>::zeros(block_slice.n_unique());

        full_filter
            .iter()
            .zip(block_slice.index_map.iter())
            .for_each(|(&b, &i)| {
                filter[i as usize] = b;

                #[cfg(feature = "obliv")]
                {
                    weights[i as usize] += b.as_u32();
                }

                #[cfg(not(feature = "obliv"))]
                {
                    weights[i as usize] += b as u32;
                }
            });

        #[cfg(feature = "obliv")]
        let weights = weights.map(|v| Real::from(v.as_u64()));

        #[cfg(not(feature = "obliv"))]
        let weights = weights.map(|&v| v as Real);

        #[cfg(feature = "obliv")]
        let inv_weights = weights.map(|&v| {
            v.tp_eq(&Real::ZERO)
                .select(Real::ZERO, Real::protect_i64(1) / v)
        });

        #[cfg(not(feature = "obliv"))]
        let inv_weights = weights.map(|&v| if v == 0. { 0. } else { 1. / v });

        #[cfg(feature = "obliv")]
        let n_unique_states = filter
            .iter()
            .fold(UInt::protect(0), |acc, v| acc + v.as_u32());

        #[cfg(not(feature = "obliv"))]
        let n_unique_states = filter.iter().filter(|&&v| v).count() as u32;

        Self {
            index_map: block_slice.index_map.clone(),
            haplotypes: block_slice.haplotypes.clone(),
            weights,
            inv_weights,
            n_unique_states,
            full_filter: Array1::from_vec(full_filter.to_owned()),
        }
    }

    pub fn n_sites(&self) -> usize {
        self.haplotypes.nrows()
    }

    pub fn n_full_haps(&self) -> usize {
        self.index_map.len()
    }

    pub fn n_unique_haps(&self) -> usize {
        self.weights.len()
    }

    pub fn n_unique_states(&self) -> UInt {
        self.n_unique_states
    }

    pub fn expand_pos(&self, pos: usize) -> Array1<i8> {
        let haps = self.haplotypes.row(pos);
        self.expand(haps)
    }

    fn expand(&self, haps: ArrayView1<u8>) -> Array1<i8> {
        let haps = BitSlice::<u8, Lsb0>::from_slice(haps.as_slice().unwrap());
        let mut expanded = Array1::<i8>::zeros(self.n_unique_haps());
        for (src, tar) in haps.iter().zip(expanded.iter_mut()) {
            *tar = *src as i8;
        }
        expanded
    }
}

//#[derive(Clone)]
//pub struct FilteredBlockSlice<'a> {
//pub index_map: Array1<u16>,
//pub haplotypes: ArrayView2<'a, u8>,
//pub weights: Array1<f64>,
//pub inv_weights: Array1<f64>,
//pub n_unique: usize,
//pub filter: Array1<bool>,
//}

//impl<'a> FilteredBlockSlice<'a> {
//pub fn from_block_slice(block_slice: &BlockSlice<'a>, full_filter: &[Bool]) -> Self {
//#[cfg(feature = "obliv")]
//let full_filter = full_filter.iter().map(|v| v.expose()).collect::<Vec<_>>();

//let mut filter = Array1::<bool>::from_elem(block_slice.n_unique(), false);
//let mut weights = Array1::<usize>::zeros(block_slice.n_unique());
//full_filter
//.iter()
//.zip(block_slice.index_map.iter())
//.filter(|(&b, _)| b)
//.for_each(|(_, &i)| {
//filter[i as usize] = true;
//weights[i as usize] += 1;
//});

//let mut count = 0;
//let remap = filter
//.iter()
//.map(|&b| {
//let tmp = count;
//count += b as u16;
//tmp
//})
//.collect::<Vec<_>>();

//let new_index_map = Array1::from_iter(
//full_filter
//.iter()
//.zip(block_slice.index_map.iter())
//.filter(|(&b, _)| b)
//.map(|(_, &i)| remap[i as usize]),
//);

//let weights = Array1::from_iter(weights.into_iter().filter(|&v| v != 0).map(|v| v as f64));
//let inv_weights = weights.map(|&v| 1. / v);

//let n_unique = filter.iter().filter(|&&b| b).count();

//Self {
//index_map: new_index_map,
//haplotypes: block_slice.haplotypes.clone(),
//weights,
//inv_weights,
//n_unique,
//filter,
//}
//}

//pub fn n_sites(&self) -> usize {
//self.haplotypes.nrows()
//}

//pub fn n_full(&self) -> usize {
//self.index_map.len()
//}

//pub fn n_unique(&self) -> usize {
//self.n_unique
//}

//pub fn expand_pos(&self, pos: usize) -> Array1<i8> {
//let haps = self.haplotypes.row(pos);
//let expanded = self.expand(haps);
//Array1::from_iter(
//expanded
//.into_iter()
//.zip(self.filter.iter())
//.filter_map(|(i, &b)| if b { Some(i) } else { None }),
//)
//}

//fn expand(&self, haps: ArrayView1<u8>) -> Array1<i8> {
//let haps = BitSlice::<u8, Lsb0>::from_slice(haps.as_slice().unwrap());
//let mut expanded = Array1::<i8>::zeros(self.filter.len());
//for (src, tar) in haps.iter().zip(expanded.iter_mut()) {
//*tar = *src as i8;
//}
//expanded
//}

////pub fn get_members(&self) -> Vec<Vec<usize>> {
////let mut members = vec![Vec::new(); self.n_unique];
////for (i, &index) in self.index_map.iter().enumerate() {
////members[index].push(i);
////}
////members
////}
//}
