mod leaky;
mod linear_scan;
mod path_oram;

use mc_oblivious_traits::{
    typenum::{Unsigned, U64},
    A64Bytes,
};

pub use linear_scan::*;
pub use path_oram::*;
pub use leaky::*;

pub type BlockSize = U64;
pub type Block = A64Bytes<BlockSize>;

pub trait ORAMBackend {
    fn read_block(&mut self, block_index: usize) -> Block;
    fn write_block(&mut self, block: &Block, block_index: usize);
}

pub trait ORAMBackendCreator {
    type Backend: ORAMBackend;
    fn new_empty(&self, n_blocks: usize) -> Self::Backend;
    fn new_init<'a>(
        &self,
        blocks_iter: impl Iterator<Item = std::borrow::Cow<'a, Block>> + ExactSizeIterator,
    ) -> Self::Backend;
}

pub struct ORAM<OBC: ORAMBackendCreator, const VALUE_SIZE: usize> {
    values_per_block: usize,
    backend: OBC::Backend,
}

impl<OBC: ORAMBackendCreator, const VALUE_SIZE: usize> ORAM<OBC, VALUE_SIZE> {
    pub fn new_empty(n_values: usize, oram_backend_creator: OBC) -> Self {
        assert!(VALUE_SIZE <= 64);
        let values_per_block = BlockSize::USIZE / VALUE_SIZE;
        let n_blocks = (n_values + values_per_block - 1) / values_per_block;
        let backend = oram_backend_creator.new_empty(n_blocks);
        Self {
            values_per_block,
            backend,
        }
    }

    pub fn new_init(values: &[[u8; VALUE_SIZE]], oram_backend_creator: OBC) -> Self {
        let values_per_block = BlockSize::USIZE / VALUE_SIZE;
        let iter = values.chunks(values_per_block).map(|chunk| {
            let mut iter: Box<dyn Iterator<Item = u8>> = Box::new(std::iter::empty());
            for value in chunk {
                iter = Box::new(iter.chain(value.iter().cloned()));
            }
            if chunk.len() < values_per_block || BlockSize::USIZE % VALUE_SIZE != 0 {
                iter = Box::new(iter.chain(std::iter::repeat(0u8)));
            };
            use std::iter::FromIterator;
            let block = Block::from_iter(iter);
            std::borrow::Cow::<Block>::Owned(block)
        });
        let backend = oram_backend_creator.new_init(iter);
        Self {
            values_per_block,
            backend,
        }
    }

    // Assuming reading from a cache line is safe
    pub fn read(&mut self, index: usize) -> [u8; VALUE_SIZE] {
        let block_index = index / self.values_per_block;
        let index_in_block = index % self.values_per_block;
        let block = self.backend.read_block(block_index);
        let mut result = [0u8; VALUE_SIZE];
        result.copy_from_slice(
            &block[(index_in_block * VALUE_SIZE)..((index_in_block + 1) * VALUE_SIZE)],
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaky() {
        const VALUE_SIZE: usize = 6;
        let n_values = 100;

        let mut values = vec![[0u8; VALUE_SIZE]; n_values];
        let mut counter = 0u8;

        for value in values.iter_mut() {
            for byte in value.iter_mut() {
                *byte = counter;
                counter = counter.wrapping_add(1);
            }
        }

        let mut oram = ORAM::<_, VALUE_SIZE>::new_init(&values, LeakyORAMCreator);
        for index in 0..n_values {
            let result = oram.read(index);
            assert_eq!(result, values[index]);
        }
    }

    #[test]
    fn linear_scan() {
        const VALUE_SIZE: usize = 6;
        let n_values = 100;

        let mut values = vec![[0u8; VALUE_SIZE]; n_values];
        let mut counter = 0u8;

        for value in values.iter_mut() {
            for byte in value.iter_mut() {
                *byte = counter;
                counter = counter.wrapping_add(1);
            }
        }

        let mut oram = ORAM::<_, VALUE_SIZE>::new_init(&values, LinearScanningORAMCreator);
        for index in 0..n_values {
            let result = oram.read(index);
            assert_eq!(result, values[index]);
        }
    }

    #[test]
    fn path_oram() {
        const VALUE_SIZE: usize = 6;
        let n_values = 100;

        let mut values = vec![[0u8; VALUE_SIZE]; n_values];
        let mut counter = 0u8;

        for value in values.iter_mut() {
            for byte in value.iter_mut() {
                *byte = counter;
                counter = counter.wrapping_add(1);
            }
        }

        let mut oram =
            ORAM::<_, VALUE_SIZE>::new_init(&values, PathORAMCreator::with_stash_size(11));
        for index in 0..n_values {
            let result = oram.read(index);
            assert_eq!(result, values[index]);
        }
    }

}
