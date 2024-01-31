use criterion::{black_box, criterion_group, criterion_main, Criterion};
//use obliv_utils::vec::OblivVec;
use timing_shield::{TpBool, TpU64};

//fn cond_copy_from(vec1: &mut OblivVec<TpU64>, vec2: &OblivVec<TpU64>) {
//vec1.cond_copy_from(&vec2, TpBool::protect(true));
//}

//fn cond_copy_from_slice(vec: &mut OblivVec<TpU64>, slice: &[TpU64]) {
//vec.cond_copy_from_slice(slice, TpBool::protect(true));
//}

//fn cond_copy_from_benchmark(c: &mut Criterion) {
//let n = 100000;
//let mut vec1 = OblivVec::<TpU64>::new();
//let mut vec2 = OblivVec::<TpU64>::new();
//for i in 0..n {
//vec1.push(TpU64::protect(i));
//}
//for i in (0..n).rev() {
//vec2.push(TpU64::protect(i));
//}

//c.bench_function("cond_copy_from", |b| {
//b.iter(|| cond_copy_from(black_box(&mut vec1), black_box(&vec2)))
//});
//}

//fn cond_copy_from_slice_benchmark(c: &mut Criterion) {
//let n = 100000;
//let mut vec1 = OblivVec::<TpU64>::new();
//let mut vec2 = Vec::<TpU64>::new();
//for i in 0..n {
//vec1.push(TpU64::protect(i));
//}
//for i in (0..n).rev() {
//vec2.push(TpU64::protect(i));
//}

//c.bench_function("cond_copy_from_slice", |b| {
//b.iter(|| cond_copy_from_slice(black_box(&mut vec1), black_box(&vec2)))
//});
//}

fn compare_cond_copy_benchmark(c: &mut Criterion) {
    let n = 100000;

    let src = (0..n).map(|v| TpU64::protect(v as u64)).collect::<Vec<_>>();

    let mut group = c.benchmark_group("Conditional copy");
    group.bench_function("cond_copy_slice_1 true", |b| {
        let mut tar = vec![TpU64::protect(0); n];
        b.iter(|| {
            obliv_utils::cond_copy::cond_copy_slice_1(
                black_box(&mut tar),
                black_box(&src),
                black_box(TpBool::protect(true)),
            )
        });
    });
    group.bench_function("cond_copy_slice_1 false", |b| {
        let mut tar = vec![TpU64::protect(0); n];
        b.iter(|| {
            obliv_utils::cond_copy::cond_copy_slice_1(
                black_box(&mut tar),
                black_box(&src),
                black_box(TpBool::protect(false)),
            )
        });
    });
    group.bench_function("cond_copy_slice_2", |b| {
        let mut tar = vec![TpU64::protect(0); n];
        b.iter(|| {
            obliv_utils::cond_copy::cond_copy_slice_2(
                black_box(&mut tar),
                black_box(&src),
                black_box(TpBool::protect(true)),
            )
        });
    });
    group.bench_function("cond_copy_slice_3", |b| {
        let mut tar = vec![TpU64::protect(0); n];
        b.iter(|| {
            obliv_utils::cond_copy::cond_copy_slice_3(
                black_box(&mut tar),
                black_box(&src),
                black_box(TpBool::protect(true)),
            )
        });
    });
    group.bench_function("cond_copy_slice_4", |b| {
        let mut tar = vec![TpU64::protect(0); n];
        b.iter(|| {
            obliv_utils::cond_copy::cond_copy_slice_4(
                black_box(&mut tar),
                black_box(&src),
                black_box(TpBool::protect(true)),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, compare_cond_copy_benchmark,);
criterion_main!(benches);
