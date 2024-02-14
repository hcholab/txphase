use compressed_pbwt::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn pbwt(c: &mut Criterion) {
    let path = "tests/test.vcf";

    let (full_blocks, _) = full::full_ref_blocks_from_path(path);

    let compressed = full_blocks
        .iter()
        .map(|v| compressed::Compressed::new(v))
        .collect::<Vec<_>>();

    let f1 = |full_blocks: &Vec<full::FullBlock>| {
        let n_haps = full_blocks[0].n_haps();
        let mut cur_ppa = (0..n_haps).collect::<Vec<_>>();
        let mut cur_div = vec![0usize; n_haps];
        for b in full_blocks {
            (cur_ppa, cur_div) = b.pbwt(&cur_ppa, &cur_div);
        }
    };

    let f2 = |compressed: &Vec<compressed::Compressed>| {
        for block in compressed {
            crate::pbwt_trie::PbwtTrie::transform(
                block.start_site,
                block.members.to_owned(),
                block.iter(),
                block.n_uniques(),
                block.n_sites,
            );
        }
    };

    let mut group = c.benchmark_group("PBWT");
    group.bench_function("Full", |b| b.iter(|| f1(black_box(&full_blocks))));
    group.bench_function("Compressed", |b| b.iter(|| f2(black_box(&compressed))));

    group.finish();
}

pub fn top_4_neighbors(c: &mut Criterion) {
    let path = "tests/test.vcf";

    let (full_blocks, _) = full::full_ref_blocks_from_path(path);
    let full_ref_haps = full::full_ref_from_path(path);
    let input_hap = full_ref_haps[0].to_owned();
    let n_haps = full_blocks[0].n_haps();

    #[cfg(feature = "obliv")]
    let input_hap = input_hap
        .into_iter()
        .map(|v| timing_shield::TpBool::protect(v))
        .collect::<Vec<_>>();

    let compressed = full_blocks
        .iter()
        .map(|v| compressed::Compressed::new(v))
        .collect::<Vec<_>>();

    let pbwt = compressed
        .iter()
        .map(|block| {
            crate::pbwt_trie::PbwtTrie::transform(
                block.start_site,
                block.members.to_owned(),
                block.iter(),
                block.n_uniques(),
                block.n_sites,
            )
        })
        .collect::<Vec<_>>();

    let input_hap = compressed
        .iter()
        .map(|block| &input_hap[block.start_site..block.start_site + block.n_sites])
        .collect::<Vec<_>>();

    let f = |pbwt: &Vec<pbwt_trie::PbwtTrie>, input_hap: &Vec<&[Bool]>| {
        #[cfg(feature = "obliv")]
        let mut expanded_div = vec![timing_shield::TpU64::protect(0); n_haps];

        #[cfg(not(feature = "obliv"))]
        let mut expanded_div = vec![0; n_haps];

        for (pbwt, input_hap) in pbwt.into_iter().zip(input_hap.into_iter()) {
            pbwt.find_top_neighbors(input_hap, &mut expanded_div, 4);
        }
    };

    c.bench_function("Top 4 Neighbors", |b| {
        b.iter(|| f(black_box(&pbwt), black_box(&input_hap)))
    });
}

pub fn neighbors_set(c: &mut Criterion) {
    let path = "tests/test.vcf";

    let (full_blocks, _) = full::full_ref_blocks_from_path(path);
    let full_ref_haps = full::full_ref_from_path(path);
    let input_hap = full_ref_haps[0].to_owned();
    let n_haps = full_blocks[0].n_haps();

    #[cfg(feature = "obliv")]
    let input_hap = input_hap
        .into_iter()
        .map(|v| timing_shield::TpBool::protect(v))
        .collect::<Vec<_>>();

    let compressed = full_blocks
        .iter()
        .map(|v| compressed::Compressed::new(v))
        .collect::<Vec<_>>();

    let pbwt = compressed
        .iter()
        .map(|block| {
            crate::pbwt_trie::PbwtTrie::transform(
                block.start_site,
                block.members.to_owned(),
                block.iter(),
                block.n_uniques(),
                block.n_sites,
            )
        })
        .collect::<Vec<_>>();

    let f = |input_hap: &[Bool], pbwt: &Vec<pbwt_trie::PbwtTrie>| {
        neighbors::find_neighbors_set(input_hap, pbwt, 4, n_haps);
    };

    c.bench_function("4 Neighbors Set", |b| {
        b.iter(|| f(black_box(&input_hap), black_box(&pbwt)))
    });
}

criterion_group!(benches, pbwt, neighbors_set);

criterion_main!(benches);
