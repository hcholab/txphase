//mod leaky;
mod linear_scan;
//mod path_oram;

//pub use leaky::*;
pub use linear_scan::*;
//pub use path_oram::*;
use crate::align::A64Bytes;

use timing_shield::TpU32;

pub trait BaseORAMBackend<const N: usize> {
    fn len(&self) -> usize;
}

pub trait ReadOnlyORAMBackend<const N: usize>: BaseORAMBackend<N> {
    fn read_block(&mut self, block_index: TpU32) -> Box<A64Bytes<N>>;
    fn batch_read_blocks(&mut self, block_indices: &[TpU32]) -> Vec<A64Bytes<N>>;
}

pub trait WriteOnlyORAMBackend<const N: usize>: BaseORAMBackend<N> {
    fn write_block(&mut self, block: &A64Bytes<N>, block_index: TpU32);
    fn batch_write_blocks(&mut self, blocks: &[A64Bytes<N>], block_indices: &[TpU32]);
    fn fast_modify_block_with(&mut self, block_index: TpU32, func: impl Fn(&mut A64Bytes<N>));
    fn batch_fast_modify_blocks_with(
        &mut self,
        block_indices: &[TpU32],
        funcs: &[impl Fn(&mut A64Bytes<N>)],
    );
    fn into_iter(self) -> Box<dyn Iterator<Item = A64Bytes<N>>>;
}

pub trait BaseORAMBackendCreator<const N: usize> {
    type ORAM: BaseORAMBackend<N>;
    fn new(&self, n_blocks: u32) -> Self::ORAM;
}

pub trait ReadOnlyORAMBackendCreator<const N: usize>: BaseORAMBackendCreator<N>
where
    Self::ORAM: ReadOnlyORAMBackend<N>,
{
    fn from_iter<'a>(
        &self,
        blocks_iter: impl Iterator<Item = std::borrow::Cow<'a, A64Bytes<N>>> + ExactSizeIterator,
    ) -> Self::ORAM;
}

pub trait WriteOnlyORAMBackendCreator<const N: usize>: BaseORAMBackendCreator<N>
where
    Self::ORAM: WriteOnlyORAMBackend<N>,
{
}

pub trait RWORAMBackendCreator<const N: usize>:
    ReadOnlyORAMBackendCreator<N> + WriteOnlyORAMBackendCreator<N>
where
    Self::ORAM: ReadOnlyORAMBackend<N> + WriteOnlyORAMBackend<N>,
{
}
