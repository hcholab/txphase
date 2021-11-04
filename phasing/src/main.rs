#![feature(llvm_asm)]
#![allow(dead_code)]
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

use genotype_graph::GenotypeGraph;
use ndarray::{Array1, Array2, Array3, ArrayView2};
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::str::FromStr;
use std::time::Instant;
use crate::genotype_graph::TransProbOption;
const HOST_PORT: u16 = 1234;

struct HmmParams {
    pub eprob: Real,
    pub rev_eprob: Real,
    pub rprob: Real,
    pub rev_rprob: Real,
}

impl HmmParams {
    pub fn init(eprob: f32, rprob: f32) -> Self {
        let rev_rprob = 1. - rprob;
        let rev_eprob = 1. - eprob;
        #[cfg(feature = "leak-resist")]
        let (rprob, rev_rprob, eprob, rev_eprob) = (
            Real::protect_f32(rprob),
            Real::protect_f32(rev_rprob),
            Real::protect_f32(eprob),
            Real::protect_f32(rev_eprob),
        );
        Self {
            eprob,
            rev_eprob,
            rprob,
            rev_rprob,
        }
    }
}

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
    let t = t
        .into_iter()
        .map(|g| Genotype::protect(g))
        .collect::<Vec<_>>();

    let t = Array1::<Genotype>::from_vec(t);
    let missing_bitmask = vec![false; ref_panel_meta.n_markers];

    assert_eq!(ref_panel_meta.n_markers, t.len());

    let s = 4;
    let capacity = 400;
    // hmm parameters
    let ref_eprob = 0.01; // error
    let ref_rprob = 0.05; // recombination
    let hmm_params = HmmParams::init(ref_eprob, ref_rprob);

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
    let mut phased = initialize::initialize(
        row_iter,
        t.view(),
        &missing_bitmask,
        ref_panel_meta.n_markers,
        ref_panel_meta.n_haps,
    );

    println!("Initialization: {} ms", (Instant::now() - now).as_millis());
    println!("",);
    println!("",);

    phased = burnin_iter(
        phased.view(),
        &geno_graph,
        &ref_panel_meta,
        &ref_panel_blocks,
        s,
        capacity,
        &hmm_params,
        &mut rng,
    );

    phased = pruning_iter(
        phased.view(),
        &mut geno_graph,
        &ref_panel_meta,
        &ref_panel_blocks,
        s,
        capacity,
        &hmm_params,
        &mut rng,
    );


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
        let (estm, tprob) = main_iter(
            phased.view(),
            &geno_graph,
            &ref_panel_meta,
            &ref_panel_blocks,
            s,
            capacity,
            &hmm_params,
            &mut rng,
        );

        // Add tprob to aggregator
        if i == 0 {
            tprob_agg = tprob;
        } else {
            tprob_agg += &tprob;
        }
        phased = estm;
    }

    let now = Instant::now();
    let phase_ind = crate::viterbi::viterbi(tprob_agg.view());
    println!("Viterbi: {} ms", (Instant::now() - now).as_millis());
    phased = geno_graph.get_haps(phase_ind.view());

}

fn burnin_iter(
    estimated_haps: ArrayView2<Genotype>,
    genotype_graph: &GenotypeGraph,
    ref_panel_meta: &m3vcf::RefPanelMeta,
    ref_panel_blocks: &[m3vcf::Block],
    s: usize,
    capacity: usize,
    hmm_params: &HmmParams,
    mut rng: impl rand::Rng,
) -> Array2<Genotype> {
    println!("=== Burn-in iteration ===",);
    let now = Instant::now();
    let filtered_ref_panel = update_ref_panel(
        ref_panel_blocks,
        estimated_haps,
        ref_panel_meta.n_markers,
        ref_panel_meta.n_haps,
        s,
        capacity,
    );

    let (phase_ind, _) = genotype_graph.forward_sampling(
        filtered_ref_panel.view(),
        hmm_params.rprob,
        hmm_params.rev_rprob,
        hmm_params.eprob,
        hmm_params.rev_eprob,
        &mut rng,
        TransProbOption::Burnin,
    );
    let out = genotype_graph.get_haps(phase_ind.view());
    println!(
        "Burn-in iteration: {} ms",
        (Instant::now() - now).as_millis()
    );
    println!("",);

    out
}

fn pruning_iter(
    estimated_haps: ArrayView2<Genotype>,
    genotype_graph: &mut GenotypeGraph,
    ref_panel_meta: &m3vcf::RefPanelMeta,
    ref_panel_blocks: &[m3vcf::Block],
    s: usize,
    capacity: usize,
    hmm_params: &HmmParams,
    mut rng: impl rand::Rng,
) -> Array2<Genotype> {
    println!("=== Pruning iteration ===",);
    let now = Instant::now();
    let filtered_ref_panel = update_ref_panel(
        &ref_panel_blocks,
        estimated_haps,
        ref_panel_meta.n_markers,
        ref_panel_meta.n_haps,
        s,
        capacity,
    );
    let (phase_ind, tprob) = genotype_graph.forward_sampling(
        filtered_ref_panel.view(),
        hmm_params.rprob,
        hmm_params.rev_rprob,
        hmm_params.eprob,
        hmm_params.rev_eprob,
        &mut rng,
        TransProbOption::Prune,
    );

    //println!("{:?}", tprob.as_ref().unwrap());

    let out = genotype_graph.get_haps(phase_ind.view());
    genotype_graph.prune(tprob.unwrap().view());
    println!(
        "Pruning iteration: {} ms",
        (Instant::now() - now).as_millis()
    );
    println!("",);
    out
}


fn main_iter(
    estimated_haps: ArrayView2<Genotype>,
    genotype_graph: &GenotypeGraph,
    ref_panel_meta: &m3vcf::RefPanelMeta,
    ref_panel_blocks: &[m3vcf::Block],
    s: usize,
    capacity: usize,
    hmm_params: &HmmParams,
    mut rng: impl rand::Rng,
) -> (Array2<Genotype>, Array3<Real>) {
    println!("=== Main iteration ===");
    let now = Instant::now();
    let filtered_ref_panel = update_ref_panel(
        ref_panel_blocks,
        estimated_haps,
        ref_panel_meta.n_markers,
        ref_panel_meta.n_haps,
        s,
        capacity,
    );

    let (phase_ind, tprob) = genotype_graph.forward_sampling(
        filtered_ref_panel.view(),
        hmm_params.rprob,
        hmm_params.rev_rprob,
        hmm_params.eprob,
        hmm_params.rev_eprob,
        &mut rng,
        TransProbOption::Main,
    );

    let out = genotype_graph.get_haps(phase_ind.view());
    println!("Main iteration: {} ms", (Instant::now() - now).as_millis());
    println!("",);

    (out, tprob.unwrap())
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
        let (filtered, n_filtered) =
            oram_sgx::obliv_filter(&bitmap[..], &transposed, capacity as u32);

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
