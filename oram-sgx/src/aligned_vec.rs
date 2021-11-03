use crate::align::A64Bytes;

pub struct AlignedVec<const N: usize> {
    inner: Vec<A64Bytes<N>>,
    len: usize,
}
