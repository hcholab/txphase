#![feature(int_log)]
#![allow(dead_code)]
mod genotype_graph;
mod hmm;
mod mcmc;
mod neighbors_finding;
mod oram;
mod pbwt;
mod ref_panel;
mod union_filter;
mod utils;
mod windows_split;
mod variants;

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

use crate::mcmc::IterOption;
use ndarray::Array1;
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::str::FromStr;

const HOST_PORT: u16 = 1234;

fn main() {
    //let min_window_len_cm = 2.5;
    //let min_window_len_cm = 4.0;
    let min_window_len_cm = 6.0;
    let pbwt_modulo = 0.02;
    let n_pos_window_overlap = 10;
    let s = 4;

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
    let sites_bitmask: Vec<bool> = bincode::deserialize_from(&mut host_stream).unwrap();

    let (ref_panel_new, afreqs) =
        ref_panel::m3vcf_scan(&ref_panel_meta, &ref_panel_blocks, &sites_bitmask);

    let cms = {
        let cms: Vec<f64> = bincode::deserialize_from(&mut host_stream).unwrap();
        cms.into_iter()
            .zip(sites_bitmask.iter())
            .filter(|(_, b)| **b)
            .map(|(cm, _)| cm)
            .collect::<Vec<_>>()
    };
    println!("#sites = {}", cms.len());

    let mcmc_params = mcmc::McmcSharedParams::new(
        ref_panel_new,
        cms,
        afreqs,
        min_window_len_cm,
        n_pos_window_overlap,
        pbwt_modulo,
        s,
    );

    let genotypes: Vec<i8> = bincode::deserialize_from(&mut host_stream).unwrap();

    //#[cfg(feature = "leak-resist")]
    //let genotypes = genotypes
    //.into_iter()
    //.map(|g| Genotype::protect(g))
    //.collect::<Vec<_>>();

    let genotypes = Array1::<Genotype>::from_vec(genotypes);

    let mut rng = rand::thread_rng();
    //use rand::SeedableRng;
    //let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

    let mut mcmc = mcmc::Mcmc::initialize(&mcmc_params, genotypes.view());

    for _ in 0..6 {
        mcmc.iteration(IterOption::Burnin, &mut rng);
    }

    for _ in 0..2 {
        mcmc.iteration(IterOption::Pruning, &mut rng);
        mcmc.iteration(IterOption::Burnin, &mut rng);
    }

    let phased = mcmc.main_finalize(5, rng);



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
