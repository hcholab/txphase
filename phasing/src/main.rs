#![allow(dead_code)]
#![feature(llvm_asm)]
#![feature(destructuring_assignment)]
mod block;
mod genotype_graph;
mod initialize;
mod neighbors_finding;
mod oram;
mod pbwt;
mod union_filter;
mod utils;
mod viterbi;

#[cfg(feature = "leak-resist")]
mod inner {
    use tp_fixedpoint::timing_shield::{TpBool, TpI32, TpI8, TpU32, TpU8};
    pub type Genotype = TpI8;
    pub type UInt = TpU32;
    pub type Int = TpI32;
    pub type U8 = TpU8;
    pub type Bool = TpBool;
    #[cfg(not(feature = "leak-resist-fast"))]
    pub type Real = tp_fixedpoint::TpLnFixed<20>;
    #[cfg(feature = "leak-resist-fast")]
    pub type Real = tp_fixedpoint::TpFixed64<30>;
}

#[cfg(not(feature = "leak-resist"))]
mod inner {
    pub type Genotype = i8;
    pub type UInt = u32;
    pub type Int = i32;
    pub type U8 = u8;
    pub type Bool = bool;
    pub type Real = f32;
}

use inner::*;

use std::net::{IpAddr, SocketAddr, TcpListener};
use std::str::FromStr;
const HOST_PORT: u16 = 1234;

fn main() {
    let (host_stream, _host_socket) = TcpListener::bind(SocketAddr::from((
        IpAddr::from_str("127.0.0.1").unwrap(),
        HOST_PORT,
    )))
    .unwrap()
    .accept()
    .unwrap();

    let mut host_stream = bufstream::BufStream::new(host_stream);

    let ref_panel_meta: m3vcf::RefPanelMeta = bincode::deserialize_from(&mut host_stream).unwrap();
    let ref_panel_blocks: Vec<m3vcf::Block> = bincode::deserialize_from(&mut host_stream).unwrap();
    let t: Vec<i8> = bincode::deserialize_from(&mut host_stream).unwrap();

    #[cfg(feature = "leak-resist")]
    let t = t.into_iter().map(|g| Genotype::protect(g)).collect::<Vec<_>>();

    let t = Array1::<Genotype>::from_vec(t);
    let missing_bitmask = vec![false; ref_panel_meta.n_markers];

    assert_eq!(ref_panel_meta.n_markers, t.len());

    let s = 4;
    let capacity = 400;
    // hmm parameters
    let ref_rprob = 0.05; // recombination
    let ref_eprob = 0.01; // error
    let rprob = ref_rprob; // recombination
    let eprob = ref_eprob; // error
    let rev_rprob = 1. - rprob; // recombination
    let rev_eprob = 1. - eprob; // error
    #[cfg(feature = "leak-resist")]
    let (rprob, rev_rprob, eprob, rev_eprob) = (
        Real::protect_f32(rprob),
        Real::protect_f32(rev_rprob),
        Real::protect_f32(eprob),
        Real::protect_f32(rev_eprob),
    );

    let mut rng = rand::thread_rng();

    let now = Instant::now();
    let mut geno_graph = GenotypeGraph::build(t.view());
    println!(
        "Build genotype graph: {} ms",
        (Instant::now() - now).as_millis()
    );

    println!("=== Initialization ===",);
    let now = Instant::now();
    let row_iter = block::RowIterator::from_blocks_iter(ref_panel_blocks.iter());
    let phased = initialize::initialize(
        row_iter,
        t.view(),
        &missing_bitmask,
        ref_panel_meta.n_markers,
        ref_panel_meta.n_haps,
    );

    println!("Initialization: {} ms", (Instant::now() - now).as_millis());
    println!("",);
    println!("",);

    println!("=== Burn-in iteration ===",);
    let now = Instant::now();
    let filtered_ref_panel = update_ref_panel(
        &ref_panel_blocks,
        phased.view(),
        ref_panel_meta.n_markers,
        ref_panel_meta.n_haps,
        s,
        capacity,
    );
    let now_h = Instant::now();
    let (phase_ind, _) = geno_graph.forward_sampling(
        filtered_ref_panel.view(),
        rprob,
        rev_rprob,
        eprob,
        rev_eprob,
        &mut rng,
        0,
    );
    println!("HMM: {} ms", (Instant::now() - now_h).as_millis());
    let phased = geno_graph.get_haps(phase_ind.view());
    println!(
        "Burn-in iteration: {} ms",
        (Instant::now() - now).as_millis()
    );
    println!("",);
    println!("",);


    println!("=== Pruning iteration ===",);
    let now = Instant::now();
    let filtered_ref_panel = update_ref_panel(
        &ref_panel_blocks,
        phased.view(),
        ref_panel_meta.n_markers,
        ref_panel_meta.n_haps,
        s,
        capacity,
    );
    let now_h = Instant::now();
    let (phase_ind, tprob) = geno_graph.forward_sampling(
        filtered_ref_panel.view(),
        rprob,
        rev_rprob,
        eprob,
        rev_eprob,
        &mut rng,
        2,
    );
    println!("HMM: {} ms", (Instant::now() - now_h).as_millis());
    let mut phased = geno_graph.get_haps(phase_ind.view());
    geno_graph.prune(tprob.unwrap().view());
    println!(
        "Pruning iteration: {} ms",
        (Instant::now() - now).as_millis()
    );
    println!("",);
    println!("",);

    {
        #[cfg(feature = "leak-resist")]
        let phased = Array2::from_shape_fn(phased.dim(), |(i, j)| phased[[i, j]].expose());
        bincode::serialize_into(&mut host_stream, &phased).unwrap();
    }
    return;


    let num_main_iter = 2;
    let p = 1 << genotype_graph::HET_PER_BLOCK;
    let mut tprob_agg = unsafe { Array3::uninit((ref_panel_meta.n_markers, p, p)).assume_init() };
    for i in 0..num_main_iter {
        println!("=== Main iteration {} ===", i);
        let now = Instant::now();
        let filtered_ref_panel = update_ref_panel(
            &ref_panel_blocks,
            phased.view(),
            ref_panel_meta.n_markers,
            ref_panel_meta.n_haps,
            s,
            capacity,
        );
        let now_h = Instant::now();
        let (phase_ind, tprob) = geno_graph.forward_sampling(
            filtered_ref_panel.view(),
            rprob,
            rev_rprob,
            eprob,
            rev_eprob,
            &mut rng,
            1,
        );
        println!("HMM: {} ms", (Instant::now() - now_h).as_millis());
        phased = geno_graph.get_haps(phase_ind.view());

        // Add tprob to aggregator
        if i == 0 {
            tprob_agg = tprob.unwrap();
        } else {
            tprob_agg += &tprob.unwrap();
        }
        println!(
            "Main iteration {}: {} ms",
            i,
            (Instant::now() - now).as_millis()
        );
        println!("",);
        println!("",);
    }

    let now = Instant::now();
    let phase_ind = crate::viterbi::viterbi(tprob_agg.view());
    println!("Viterbi: {} ms", (Instant::now() - now).as_millis());
    let _phased = geno_graph.get_haps(phase_ind.view());
}

use genotype_graph::GenotypeGraph;
use ndarray::{Array1, Array2, Array3, ArrayView2};
use rand::{distributions::Distribution, thread_rng, Rng};
use std::time::Instant;

fn hmm() {
    let mut rng = rand::thread_rng();

    let m = 100; // num of variants
    let n = 1000; // num of samples

    // hmm parameters
    let ref_rprob = 0.05; // recombination
    let ref_eprob = 0.01; // error
    let rprob = ref_rprob; // recombination
    let eprob = ref_eprob; // error
    let rev_rprob = 1. - rprob; // recombination
    let rev_eprob = 1. - eprob; // error
    #[cfg(feature = "leak-resist")]
    let (rprob, rev_rprob, eprob, rev_eprob) = (
        Real::protect_f32(rprob),
        Real::protect_f32(rev_rprob),
        Real::protect_f32(eprob),
        Real::protect_f32(rev_eprob),
    );

    let mut x = unsafe { Array2::<i8>::uninit((m, n)).assume_init() };
    let mut t = unsafe { Array1::<Genotype>::uninit(m).assume_init() }; // target

    for v in x.iter_mut() {
        *v = rng.gen_range(0..2);
    }

    // Simulate haplotypes from x then combine to get example target
    {
        let mut thap = Array2::<i8>::zeros((2, m));
        for hap in 0..2 {
            thap.row_mut(hap).assign(&sample_from_hmm(
                x.view(),
                ref_rprob,
                ref_eprob,
                &mut thread_rng(),
            ));
        }
        for i in 0..m {
            #[cfg(feature = "leak-resist")]
            {
                t[i] = Genotype::protect(thap[[0, i]] + thap[[1, i]]);
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                t[i] = thap[[0, i]] + thap[[1, i]];
            }
        }
    }

    // Geno graph for target
    let now = Instant::now();
    let mut geno_graph = GenotypeGraph::build(t.view());
    println!(
        "Build genotype graph: {} ms",
        (Instant::now() - now).as_millis()
    );

    // TODO update ref
    #[cfg(feature = "leak-resist")]
    let x = {
        let mut _x = unsafe { Array2::<Genotype>::uninit((m, n)).assume_init() };
        for (a, &b) in _x.iter_mut().zip(x.iter()) {
            *a = Genotype::protect(b);
        }
        _x
    };

    println!("=== Burn-in iteration ===",);
    let now = Instant::now();
    let (phase_ind, _) =
        geno_graph.forward_sampling(x.view(), rprob, rev_rprob, eprob, rev_eprob, &mut rng, 0);
    let mut _phased = geno_graph.get_haps(phase_ind.view());
    println!(
        "Burn-in iteration: {} ms",
        (Instant::now() - now).as_millis()
    );
    println!("",);
    println!("",);

    println!("=== Pruning iteration ===",);
    let now = Instant::now();
    //TODO update ref
    let (phase_ind, tprob) =
        geno_graph.forward_sampling(x.view(), rprob, rev_rprob, eprob, rev_eprob, &mut rng, 2);
    _phased = geno_graph.get_haps(phase_ind.view());
    geno_graph.prune(tprob.unwrap().view());
    println!(
        "Pruning iteration: {} ms",
        (Instant::now() - now).as_millis()
    );
    println!("",);
    println!("",);

    let num_main_iter = 2;
    let p = 1 << genotype_graph::HET_PER_BLOCK;
    let mut tprob_agg = unsafe { Array3::uninit((m, p, p)).assume_init() };
    for i in 0..num_main_iter {
        println!("=== Main iteration {} ===", i);
        let now = Instant::now();
        //update_ref_panel(&x, &phased);
        let (phase_ind, tprob) =
            geno_graph.forward_sampling(x.view(), rprob, rev_rprob, eprob, rev_eprob, &mut rng, 1);

        _phased = geno_graph.get_haps(phase_ind.view());

        // Add tprob to aggregator
        if i == 0 {
            tprob_agg = tprob.unwrap();
        } else {
            tprob_agg += &tprob.unwrap();
        }
        println!(
            "Main iteration {}: {} ms",
            i,
            (Instant::now() - now).as_millis()
        );
        println!("",);
        println!("",);
    }

    let now = Instant::now();
    // Maximum a posteriori decoding using averaged trans probs
    let phase_ind = crate::viterbi::viterbi(tprob_agg.view());
    println!("Viterbi: {} ms", (Instant::now() - now).as_millis());
    _phased = geno_graph.get_haps(phase_ind.view());
}

// Utility function for simulating target sequences for testing purposes
fn sample_from_hmm(x: ArrayView2<i8>, rprob: f32, eprob: f32, rng: &mut impl Rng) -> Array1<i8> {
    let m = x.nrows();
    let n = x.ncols();

    let mut hap = Array1::<i8>::zeros(m);

    let mut recvec = vec![0.0; 2];
    recvec[0] = 1.0 - rprob;
    recvec[1] = rprob;

    let mut errvec = vec![0.0; 2];
    errvec[0] = 1.0 - eprob;
    errvec[1] = eprob;

    let mut z = rng.gen_range(0..n);
    hap[0] = (x[[0, z]] + sample(errvec.as_slice(), rng) as i8) % 2;

    for i in 1..m {
        let rec_flag = sample(recvec.as_slice(), rng);
        if rec_flag == 1 {
            z = rng.gen_range(0..n);
        }
        hap[i] = (x[[i, z]] + sample(errvec.as_slice(), rng) as i8) % 2;
    }

    hap
}

fn sample(weights: &[f32], rng: &mut impl Rng) -> usize {
    let dist = rand::distributions::WeightedIndex::new(&*weights).unwrap();
    dist.sample(rng)
}

fn update_ref_panel(
    blocks: &[m3vcf::Block],
    t: ArrayView2<Genotype>,
    npos: usize,
    nhap: usize,
    s: usize,
    capacity: usize,
) -> Array2<Genotype> {
    const N: usize = 1000;

    let mut filterd_ref_panel = Array2::<Genotype>::from_elem((npos, nhap), tp_value!(0, i8));
    let row_iter = block::RowIterator::from_blocks_iter(blocks.iter());
    let neighbors = neighbors_finding::find_neighbors(row_iter, t, npos, nhap, s);

    #[cfg(feature = "leak-resist")]
    let bitmap = {
        let mut bitmap = union_filter::OblivBitmap::new(nhap, oram_sgx::LinearScanningORAMCreator);
        for i in neighbors.into_iter().flatten().flatten() {
            bitmap.set(i);
        }

        bitmap
            .into_iter()
            .map(|v| tp_value!(v, bool))
            .collect::<Vec<_>>()
    };

    #[cfg(not(feature = "leak-resist"))]
    let bitmap = {
        let mut bitmap = vec![false; nhap];
        for i in neighbors.into_iter().flatten().flatten() {
            bitmap[i as usize] = true;
        }
        bitmap
    };

    let mut cur_pos = 0;
    for (f, block) in blocks.iter().enumerate() {
        let nskip = if f == 0 { 0 } else { 1 };
        let nvar = block.nvar - nskip;
        let transposed = block::block_to_aligned_transposed::<N>(block, nhap);

        #[cfg(feature = "leak-resist")]
        let (filtered, n_filtered) = oram_sgx::obliv_filter(&bitmap[..], &transposed, capacity as u32);

        #[cfg(feature = "leak-resist")]
        println!("n_filtered: {}", n_filtered.expose());

        #[cfg(not(feature = "leak-resist"))]
        let filtered = transposed
            .into_iter()
            .zip(bitmap.iter())
            .filter(|(_, b)| **b)
            .map(|(v, _)| v)
            .take(capacity)
            .collect::<Vec<_>>();

        // transpose
        for (i, hap) in filtered.into_iter().enumerate() {
            for (mut row, &geno) in filterd_ref_panel
                .slice_mut(ndarray::s![cur_pos..(cur_pos + nvar), ..])
                .rows_mut()
                .into_iter()
                .zip(hap.as_slice().iter().skip(nskip).take(nvar))
            {
                row[i] = tp_value!(geno, i8);
            }
        }
        cur_pos += nvar;
    }

    filterd_ref_panel
}
