use crate::{Block, BlockSize, ORAMBackend, ORAMBackendCreator};
use aligned_cmov::AsAlignedChunks;
use mc_oblivious_ram::{PathORAM as InnerPathORAM, U32PositionMapCreator};
use mc_oblivious_traits::{
    subtle::ConstantTimeEq, typenum, A64Bytes, CMov, HeapORAMStorage, HeapORAMStorageCreator,
    ORAMCreator, ORAM,
};
use rand::{thread_rng, SeedableRng};
use rand_hc::Hc128Rng;
use typenum::Unsigned;

#[derive(Clone)]
pub struct PathORAMCreator {
    stash_size: usize,
}

impl PathORAMCreator {
    pub fn with_stash_size(stash_size: usize) -> Self {
        Self { stash_size }
    }
}

impl ORAMBackendCreator for PathORAMCreator {
    type Backend = PathORAM;
    fn new_empty(&self, n_blocks: usize) -> Self::Backend {
        let blocks_per_inner_block = InnerBlockSize::USIZE / BlockSize::USIZE;
        let n_inner_blocks = (n_blocks + blocks_per_inner_block - 1) / blocks_per_inner_block;
        let n_inner_blocks = n_inner_blocks.next_power_of_two();
        let mut rng_maker =
            mc_oblivious_traits::rng_maker(Hc128Rng::from_rng(thread_rng()).unwrap());
        let inner =
            MyPathORAMCreator::create(n_inner_blocks as u64, self.stash_size, &mut rng_maker);
        Self::Backend {
            inner,
            n_blocks,
            blocks_per_inner_block,
        }
    }

    fn new_init<'a>(
        &self,
        mut blocks_iter: impl Iterator<Item = std::borrow::Cow<'a, Block>> + ExactSizeIterator,
    ) -> Self::Backend {
        let n_blocks = blocks_iter.len();
        let blocks_per_inner_block = InnerBlockSize::USIZE / BlockSize::USIZE;
        let n_inner_blocks =
            (blocks_iter.len() + blocks_per_inner_block - 1) / blocks_per_inner_block;
        let n_inner_blocks = n_inner_blocks.next_power_of_two();

        let mut rng_maker =
            mc_oblivious_traits::rng_maker(Hc128Rng::from_rng(thread_rng()).unwrap());
        let mut inner =
            MyPathORAMCreator::create(n_inner_blocks as u64, self.stash_size, &mut rng_maker);

        for i in 0..n_inner_blocks {
            let mut iter: Box<dyn Iterator<Item = u8>> = Box::new(std::iter::empty());
            for _ in 0..blocks_per_inner_block {
                if let Some(block) = blocks_iter.next() {
                    let block = block.into_owned();
                    iter = Box::new(iter.chain(block.into_iter()));
                } else {
                    break;
                }
            }
            iter = Box::new(iter.chain(std::iter::repeat(0u8)));
            use std::iter::FromIterator;
            let inner_block = A64Bytes::<InnerBlockSize>::from_iter(iter);
            inner.write(i as u64, &inner_block);
        }
        Self::Backend {
            inner,
            n_blocks,
            blocks_per_inner_block,
        }
    }
}

type InnerBlockSize = typenum::U1024;

pub struct PathORAM {
    inner: <MyPathORAMCreator as ORAMCreator<InnerBlockSize, R>>::Output,
    blocks_per_inner_block: usize,
    n_blocks: usize,
}

impl ORAMBackend for PathORAM {
    fn len(&self) -> usize {
        self.n_blocks
    }

    fn read_block(&mut self, block_index: usize) -> Block {
        let inner_block_index = block_index / self.blocks_per_inner_block;
        let index_in_inner_block = block_index % self.blocks_per_inner_block;
        self.inner.access(inner_block_index as u64, |inner_block| {
            let mut temp: A64Bytes<BlockSize> = Default::default();
            for (idx, block) in
                <dyn AsAlignedChunks<aligned_cmov::A64, BlockSize>>::as_aligned_chunks(inner_block)
                    .iter()
                    .enumerate()
            {
                temp.cmov((idx as u64).ct_eq(&(index_in_inner_block as u64)), block);
            }
            temp
        })
    }

    fn write_block(&mut self, new_block: &Block, block_index: usize) {
        let inner_block_index = block_index / self.blocks_per_inner_block;
        let index_in_inner_block = block_index % self.blocks_per_inner_block;
        self.inner.access(inner_block_index as u64, |inner_block| {
            for (idx, block) in
                <dyn AsAlignedChunks<aligned_cmov::A64, BlockSize>>::as_mut_aligned_chunks(
                    inner_block,
                )
                .iter_mut()
                .enumerate()
            {
                block.cmov(
                    (idx as u64).ct_eq(&(index_in_inner_block as u64)),
                    new_block,
                );
            }
        });
    }

    fn modify_block_with(&mut self, block_index: usize, func: impl Fn(&mut Block)) {
        let inner_block_index = block_index / self.blocks_per_inner_block;
        let index_in_inner_block = block_index % self.blocks_per_inner_block;
        self.inner.access(inner_block_index as u64, |inner_block| {
            let mut temp: A64Bytes<BlockSize> = Default::default();
            for (idx, block) in
                <dyn AsAlignedChunks<aligned_cmov::A64, BlockSize>>::as_mut_aligned_chunks(
                    inner_block,
                )
                .iter_mut()
                .enumerate()
            {
                temp.cmov((idx as u64).ct_eq(&(index_in_inner_block as u64)), block);
                func(&mut temp);
                block.cmov((idx as u64).ct_eq(&(index_in_inner_block as u64)), &temp);
            }
        });
    }

    fn into_iter(mut self) -> Box<dyn Iterator<Item = Block>> {
        let n_inner_blocks =
            (self.n_blocks + self.blocks_per_inner_block - 1) / self.blocks_per_inner_block;
        let mut remaining_blocks = self.n_blocks;
        Box::new(
            (0..n_inner_blocks)
                .map(move |i| {
                    let inner_block = self.inner.read(i as u64);
                    let take_n_blocks = remaining_blocks.min(self.blocks_per_inner_block);
                    remaining_blocks -= take_n_blocks;
                    <dyn AsAlignedChunks<aligned_cmov::A64, BlockSize>>::as_aligned_chunks(
                        &inner_block,
                    )
                    .into_iter()
                    .take(take_n_blocks)
                    .cloned()
                    .collect::<Vec<_>>()
                })
                .flatten(),
        )
    }
}

type R = Hc128Rng;
type BlocksPerBucket = typenum::U4; // # blocks per bucket
type BucketSize = typenum::op!(InnerBlockSize * BlocksPerBucket);
type MetaSize = typenum::U64;

struct MyPathORAMCreator;

impl ORAMCreator<InnerBlockSize, R> for MyPathORAMCreator {
    type Output =
        InnerPathORAM<InnerBlockSize, BlocksPerBucket, HeapORAMStorage<BucketSize, MetaSize>, R>;

    fn create<M: 'static + FnMut() -> R>(
        size: u64,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Self::Output {
        InnerPathORAM::new::<
            U32PositionMapCreator<InnerBlockSize, R, Self>,
            HeapORAMStorageCreator,
            M,
        >(size, stash_size, rng_maker)
    }
}
