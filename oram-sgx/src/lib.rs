mod leaky;
mod linear_scan;
mod oram;
mod path_oram;

pub use leaky::*;
pub use linear_scan::*;
pub use oram::*;
pub use path_oram::*;

pub type BlockSize = U64;
pub type Block = A64Bytes<BlockSize>;

use mc_oblivious_traits::{typenum::U64, A64Bytes};

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

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

        let mut oram = ORAM::<_, VALUE_SIZE>::new_init(
            values.iter().map(|v| Cow::Borrowed(v)),
            LeakyORAMCreator,
        );
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

        let mut oram = ORAM::<_, VALUE_SIZE>::new_init(
            values.iter().map(|v| Cow::Borrowed(v)),
            LinearScanningORAMCreator,
        );
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

        let mut oram = ORAM::<_, VALUE_SIZE>::new_init(
            values.iter().map(|v| Cow::Borrowed(v)),
            PathORAMCreator::with_stash_size(11),
        );
        for index in 0..n_values {
            let result = oram.read(index);
            assert_eq!(result, values[index]);
        }
    }
}
