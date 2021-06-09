use crate::align::A64Bytes;
use crate::backend::*;
use timing_shield::{TpEq, TpU32};

#[derive(Clone)]
pub struct LinearScanningORAMCreator;

impl<const N: usize> BaseORAMBackendCreator<N> for LinearScanningORAMCreator {
    type ORAM = LinearScanningORAM<N>;
    fn new(&self, n_blocks: u32) -> Self::ORAM {
        Self::ORAM {
            data: vec![Default::default(); n_blocks as usize],
        }
    }
}

impl<const N: usize> ReadOnlyORAMBackendCreator<N> for LinearScanningORAMCreator {
    fn from_iter<'a>(
        &self,
        blocks_iter: impl Iterator<Item = std::borrow::Cow<'a, A64Bytes<N>>> + ExactSizeIterator,
    ) -> Self::ORAM {
        Self::ORAM {
            data: blocks_iter.map(|v| v.into_owned()).collect(),
        }
    }
}

impl<const N: usize> WriteOnlyORAMBackendCreator<N> for LinearScanningORAMCreator {}
impl<const N: usize> RWORAMBackendCreator<N> for LinearScanningORAMCreator {}

pub struct LinearScanningORAM<const N: usize> {
    data: Vec<A64Bytes<N>>,
}

impl<const N: usize> BaseORAMBackend<N> for LinearScanningORAM<N> {
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<const N: usize> ReadOnlyORAMBackend<N> for LinearScanningORAM<N> {
    fn read_block(&mut self, block_index: TpU32) -> Box<A64Bytes<N>> {
        let mut result_block = Box::new(A64Bytes::<N>::default());
        for i in 0..self.data.len() {
            let choice = block_index.tp_eq(&(i as u32));
            result_block.cmov(choice, &self.data[i]);
        }
        result_block
    }

    fn batch_read_blocks(&mut self, block_indices: &[TpU32]) -> Vec<A64Bytes<N>> {
        let mut result_blocks = vec![A64Bytes::<N>::default(); block_indices.len()];
        for i in 0..self.data.len() {
            for (index, block) in block_indices.iter().zip(result_blocks.iter_mut()) {
                let choice = index.tp_eq(&(i as u32));
                block.cmov(choice, &self.data[i]);
            }
        }
        result_blocks
    }
}

impl<const N: usize> WriteOnlyORAMBackend<N> for LinearScanningORAM<N> {
    fn write_block(&mut self, block: &A64Bytes<N>, block_index: TpU32) {
        for i in 0..self.data.len() {
            let choice = block_index.tp_eq(&(i as u32));
            self.data[i].cmov(choice, block);
        }
    }

    fn batch_write_blocks(&mut self, blocks: &[A64Bytes<N>], block_indices: &[TpU32]) {
        assert_eq!(blocks.len(), block_indices.len());
        for i in 0..self.data.len() {
            for (index, block) in block_indices.iter().zip(blocks.iter()) {
                let choice = index.tp_eq(&(i as u32));
                self.data[i].cmov(choice, block);
            }
        }
    }

    fn fast_modify_block_with(&mut self, block_index: TpU32, func: impl Fn(&mut A64Bytes<N>)) {
        let mut data = Box::new(A64Bytes::<N>::default());
        for i in 0..self.data.len() {
            *data = self.data[i].clone();
            func(data.as_mut());
            let choice = block_index.tp_eq(&(i as u32));
            self.data[i].cmov(choice, &data);
        }
    }

    fn batch_fast_modify_blocks_with(
        &mut self,
        block_indices: &[TpU32],
        funcs: &[impl Fn(&mut A64Bytes<N>)],
    ) {
        assert_eq!(block_indices.len(), funcs.len());
        let mut data = Box::new(A64Bytes::<N>::default());
        for i in 0..self.data.len() {
            for (index, func) in block_indices.iter().zip(funcs.iter()) {
                *data = self.data[i].clone();
                func(data.as_mut());
                let choice = index.tp_eq(&(i as u32));
                self.data[i].cmov(choice, &data);
            }
        }
    }

    fn into_iter(self) -> Box<dyn Iterator<Item = A64Bytes<N>>> {
        Box::new(self.data.into_iter())
    }
}
