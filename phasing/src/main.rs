#![feature(stmt_expr_attributes)]
#![allow(dead_code)]
mod genotype_graph;
mod hmm;
mod mcmc;
//mod neighbors_finding;
#[cfg(feature = "leak-resist")]
mod oram;
mod pbwt;
//mod union_filter;
mod utils;
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

#[cfg(feature = "obliv")]
mod inner {
    use tp_fixedpoint::timing_shield::{TpBool, TpI8, TpU64, TpU8};
    pub type Genotype = TpI8;
    pub type UInt = u32;
    pub type Usize = TpU64;
    pub type Int = i32;
    pub type U8 = TpU8;
    pub type Bool = TpBool;
    pub type BoolMcc = TpBool;
    pub type Real = f64;
    pub const F: usize = 52;
    pub type RealHmm = tp_fixedpoint::TpFixed64<F>;
}

#[cfg(not(feature = "obliv"))]
mod inner {
    pub type Genotype = i8;
    pub type UInt = u32;
    pub type Usize = usize;
    pub type Int = i32;
    pub type U8 = u8;
    pub type Bool = bool;
    pub type BoolMcc = bool;
    pub type Real = f64;
    pub type RealHmm = f64;
}

use inner::*;

use crate::mcmc::IterOption;
//use log::info;
use ndarray::Array1;
use rand::{RngCore, SeedableRng};
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::str::FromStr;

pub fn log_template(str1: &str, str2: &str) -> String {
    format!("\t* {str1}\t: {str2}")
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 2);
    let host_port = args[1].parse::<u16>().unwrap();

    env_logger::init();
    let min_window_len_cm = 2.5;
    //let min_window_len_cm = 3.0;
    let pbwt_modulo = 0.02;
    let min_het_rate = 0.3f64;
    let n_pos_window_overlap = (3. / min_het_rate).ceil() as usize;
    let s = 4;

    let seed = rand::thread_rng().next_u64();
    //let seed = 14907282476612848385;
    let rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    println!("Parameters:");
    println!("{}", &log_template("Seed", &format!("{seed}")));

    let (host_stream, _host_socket) = TcpListener::bind(SocketAddr::from((
        IpAddr::from_str("127.0.0.1").unwrap(),
        host_port,
    )))
    .unwrap()
    .accept()
    .unwrap();

    let mut host_stream = bufstream::BufStream::new(host_stream);

    let ref_panel_meta: m3vcf::RefPanelMeta = bincode::deserialize_from(&mut host_stream).unwrap();
    let ref_panel_blocks: Vec<m3vcf::Block> = bincode::deserialize_from(&mut host_stream).unwrap();
    let sites_bitmask: Vec<bool> = bincode::deserialize_from(&mut host_stream).unwrap();

    let (ref_panel_new, afreqs) =
        common::ref_panel::m3vcf_scan(&ref_panel_meta, &ref_panel_blocks, &sites_bitmask);

    let cms = {
        let cms: Vec<f64> = bincode::deserialize_from(&mut host_stream).unwrap();
        let mut cms = cms
            .into_iter()
            .zip(sites_bitmask.iter())
            .filter(|(_, b)| **b)
            .map(|(cm, _)| cm)
            .collect::<Vec<_>>();
        let first = cms[0];
        cms.iter_mut().for_each(|cm| *cm -= first);
        cms
    };
    println!("#sites = {}", cms.len());
    let bps = {
        let bps: Vec<u32> = bincode::deserialize_from(&mut host_stream).unwrap();
        bps.into_iter()
            .zip(sites_bitmask.iter())
            .filter(|(_, b)| **b)
            .map(|(cm, _)| cm)
            .collect::<Vec<_>>()
    };

    let genotypes: Vec<i8> = bincode::deserialize_from(&mut host_stream).unwrap();

    #[cfg(feature = "obliv")]
    let genotypes = Array1::<Genotype>::from_vec(
        genotypes
            .into_iter()
            .map(|v| Genotype::protect(v))
            .collect(),
    );

    #[cfg(not(feature = "obliv"))]
    let genotypes = Array1::<Genotype>::from_vec(genotypes);

    let mcmc_params = mcmc::McmcSharedParams::new(
        ref_panel_new,
        bps,
        cms,
        afreqs,
        min_window_len_cm,
        n_pos_window_overlap,
        pbwt_modulo,
        s,
    );

    let iterations = [
        IterOption::Burnin(5),
        IterOption::Pruning(1),
        IterOption::Burnin(1),
        IterOption::Pruning(1),
        IterOption::Burnin(1),
        IterOption::Pruning(1),
        IterOption::Main(5),
    ];
    let phased = mcmc::Mcmc::run(&mcmc_params, genotypes.view(), &iterations, rng);

    #[cfg(feature = "obliv")]
    let phased = phased.map(|v| v.expose());

    bincode::serialize_into(&mut host_stream, &phased).unwrap();
}
