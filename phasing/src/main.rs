#![allow(dead_code)]
//#![allow(unused)]
#![allow(deprecated)]
mod block;
mod genotype_graph;
mod genotypes;
mod initialize;
mod mcmc;
mod neighbors_finding;
mod oram;
mod pbwt;
mod ref_panel;
mod sampling;
mod union_filter;
mod utils;
mod viterbi;
mod windows_split;

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
    pub type Real = f64;
}

use inner::*;

use crate::mcmc::*;
use ndarray::Array1;
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::str::FromStr;

const HOST_PORT: u16 = 1234;

fn main() {
    let min_window_len_cm = 2.5;
    let n_pos_window_overlap = 10;

    let (host_stream, _host_socket) = TcpListener::bind(SocketAddr::from((
        IpAddr::from_str("127.0.0.1").unwrap(),
        HOST_PORT,
    )))
    .unwrap()
    .accept()
    .unwrap();

    let mut host_stream = bufstream::BufStream::new(host_stream);

    let ref_panel = {
        let ref_panel_meta: m3vcf::RefPanelMeta =
            bincode::deserialize_from(&mut host_stream).unwrap();

        let ref_panel_blocks: Vec<m3vcf::Block> =
            bincode::deserialize_from(&mut host_stream).unwrap();
        let sites_bitmask: Vec<bool> = bincode::deserialize_from(&mut host_stream).unwrap();
        let cms: Vec<f32> = bincode::deserialize_from(&mut host_stream).unwrap();
        ref_panel::RefPanel::new(ref_panel_meta, ref_panel_blocks, sites_bitmask, cms)
    };

    let genotypes: Vec<i8> = bincode::deserialize_from(&mut host_stream).unwrap();

    let genotypes_meta = {
        genotypes::GenotypesMeta::new()
    };

    //#[cfg(feature = "leak-resist")]
    //let genotypes = genotypes
        //.into_iter()
        //.map(|g| Genotype::protect(g))
        //.collect::<Vec<_>>();

    let genotypes = Array1::<Genotype>::from_vec(genotypes);

    let windows =
        windows_split::Windows::new(&ref_panel.cms, min_window_len_cm, n_pos_window_overlap);
    // MCMC
    let s = 4;
    let capacity = 600;
    // hmm parameters
    let eprob = 0.0001; // error
    let n_eff = 15000; //effective size of the population
    let mut rng = rand::thread_rng();
    //let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

    let mut mcmc = mcmc::Mcmc::initialize(
        &ref_panel,
        &genotypes_meta,
        genotypes.view(),
        &windows,
        eprob,
        s,
        capacity,
    );

    mcmc.iteration(IterOption::Burnin, &mut rng);
    let phased = mcmc.estimated_haps.to_owned();

    //for _ in 0..6 {
        //mcmc.iteration(IterOption::Burnin, &mut rng);
    //}

    //for _ in 0..2 {
        //mcmc.iteration(IterOption::Pruning, &mut rng);
        //mcmc.iteration(IterOption::Burnin, &mut rng);
    //}

    //let phased = mcmc.main_finalize(5, rng);
    
    bincode::serialize_into(&mut host_stream, &phased).unwrap();
}

fn analyze_graph(segment_start_markers: &[bool]) {
    let mut lens = Vec::new();
    let mut prev_len = 0;

    for (i, &b) in segment_start_markers.iter().enumerate() {
        if b {
            lens.push(i - prev_len);
            prev_len = i;
        }
    }
    lens.push(segment_start_markers.len() - prev_len);
    lens.sort();

    let avg = lens.iter().sum::<usize>() as f64 / lens.len() as f64;
    println!("agv: {}", avg);
    println!("range: {} - {}", lens[0], lens.last().unwrap());
}
