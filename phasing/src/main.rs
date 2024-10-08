#![feature(stmt_expr_attributes)]
#![feature(iter_array_chunks)]
#![feature(portable_simd)]
#![allow(dead_code)]

mod genotype_graph;
mod hmm;
mod mcmc;
mod memory;
mod neighbor_finding;
mod utils;
mod variants;

#[cfg(feature = "obliv")]
mod dynamic_fixed;

#[cfg(feature = "obliv")]
mod types {
    use tp_fixedpoint::timing_shield::{TpBool, TpI32, TpI64, TpI8, TpU16, TpU32, TpU64, TpU8};
    pub type Genotype = TpI8;
    pub type UInt = TpU32;
    pub type Usize = TpU64;
    pub type Int = TpI32;
    pub type U8 = TpU8;
    pub type I8 = TpI8;
    pub type U16 = TpU16;
    pub type U32 = TpU32;
    pub type U64 = TpU64;
    pub type I64 = TpI64;
    pub type Bool = TpBool;
    pub const F: usize = 52;
    pub type Real = tp_fixedpoint::TpFixed64<F>;
}
#[cfg(feature = "obliv")]
pub use tp_fixedpoint::timing_shield::TpEq;

#[cfg(not(feature = "obliv"))]
mod types {
    pub type U8 = u8;
    pub type I8 = i8;
    pub type U16 = u16;
    pub type U32 = u32;
    pub type U64 = u64;
    pub type Usize = usize;
    pub type Genotype = i8;
    pub type UInt = u32;
    pub type Int = i32;
    pub type Bool = bool;
    pub type Real = f64;
}

use types::*;

use crate::mcmc::IterOption;
use clap::Parser;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use std::cell::Cell;
use std::io::Write;
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::str::FromStr;

thread_local!(static WORKER_ID: Cell<usize>= Cell::new(usize::MAX));

macro_rules! log {
    ($($arg:tt)*) => {
        println!("SP Worker {}:: {}", WORKER_ID.get(), format!($($arg)*
        ));
    };
}

pub(crate) use log;

pub fn log_param_template(name: impl std::fmt::Display, param: impl std::fmt::Display) -> String {
    format!("\t* {name} : {param}")
}

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value_t = 7777)]
    host_port: u16,
    #[arg(long, default_value_t = 4.0)]
    min_window_len_cm: f64,
    #[arg(long, default_value_t = 0.1)]
    min_het_rate: f64,
    #[arg(long)]
    pbwt_depth: Option<usize>,
    #[arg(long)]
    pbwt_modulo: Option<f64>,
    #[arg(long, default_value = "5b,1p,1b,1p,1b,1p,5m")]
    mcmc_iterations: String,
    #[arg(long)]
    min_m3vcf_unique_haps: Option<usize>,
    #[arg(long)]
    max_m3vcf_unique_haps: Option<usize>,
    #[arg(short = 's', long, default_value_t = 12727004758508603152)]
    prg_seed: u64,
}

fn parse_iterations(s: &str) -> Result<Vec<IterOption>, String> {
    let mut iterations = Vec::new();
    let s = s.split_terminator(',').collect::<Vec<_>>();
    for i in s {
        if i.len() != 2 {
            return Err(format!("Invalid iterations"));
        }
        let n = i[..1].parse::<usize>().map_err(|e| format!("{e}"))?;
        if n == 0 {
            return Err(format!("Invalid iterations"));
        }
        let iteration = match i.chars().nth(1).ok_or(format!("Invalid iterations"))? {
            'b' => IterOption::Burnin(n),
            'p' => {
                if n != 1 {
                    return Err(format!("Invalid iterations"));
                }
                IterOption::Pruning(1)
            }
            'm' => IterOption::Main(n),
            _ => return Err(format!("Invalid iterations")),
        };
        iterations.push(iteration);
    }
    Ok(iterations)
}

fn main() {
    let cli = Cli::parse();

    let (host_stream, host_socket) = TcpListener::bind(SocketAddr::from((
        IpAddr::from_str("127.0.0.1").unwrap(),
        cli.host_port,
    )))
    .unwrap()
    .accept()
    .unwrap();

    let mut host_stream = bufstream::BufStream::new(host_stream);
    let worker_id: u16 = bincode::deserialize_from(&mut host_stream).unwrap();
    WORKER_ID.set(worker_id as usize);

    log!("Host connected from {}", host_socket);

    let ref_panel_meta: m3vcf::RefPanelMeta = bincode::deserialize_from(&mut host_stream).unwrap();
    let ref_panel_blocks: Vec<m3vcf::Block> = bincode::deserialize_from(&mut host_stream).unwrap();
    let sites_bitmask: Vec<bool> = bincode::deserialize_from(&mut host_stream).unwrap();
    let cms: Vec<f64> = bincode::deserialize_from(&mut host_stream).unwrap();
    let bps: Vec<u32> = bincode::deserialize_from(&mut host_stream).unwrap();

    let pbwt_depth = cli.pbwt_depth.unwrap_or(
        ((9. - ((ref_panel_meta.n_haps / 2) as f64).log10()).round() as usize)
            .min(8)
            .max(2),
    );

    let pbwt_modulo = cli.pbwt_modulo.unwrap_or(
        ((((ref_panel_meta.n_haps / 2) as f64).ln() - 50f64.ln() + 1.) * 0.01)
            .min(0.15)
            .max(0.005),
    );

    let mcmc_iterations = parse_iterations(&cli.mcmc_iterations).unwrap();

    log!("Parameters:");
    log!("{}", log_param_template("PRG seed", cli.prg_seed));
    log!("{}", log_param_template("PBWT depth", pbwt_depth));
    log!("{}", log_param_template("PBWT modulo", pbwt_modulo));
    log!(
        "{}",
        log_param_template("Min. window len. (cM)", cli.min_window_len_cm)
    );
    log!("{}", log_param_template("Min. het. rate", cli.min_het_rate));
    log!(
        "{}",
        log_param_template("MCMC iterations", cli.mcmc_iterations)
    );
    log!("");

    let n_pos_window_overlap = (3. / cli.min_het_rate).ceil() as usize;

    while let Some(genotypes) = bincode::deserialize_from::<_, Vec<i8>>(&mut host_stream).ok() {
        let sample_id: usize = bincode::deserialize_from(&mut host_stream).unwrap();
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(cli.prg_seed);

        #[cfg(feature = "obliv")]
        let genotypes = genotypes
            .into_iter()
            .map(|v| Genotype::protect(v))
            .collect::<Vec<_>>();

        let mut present = Vec::new();
        let filtered_genotypes = genotypes
            .into_iter()
            .filter(|&v| {
                #[cfg(feature = "obliv")]
                let cond = v.tp_not_eq(&-1).expose();

                #[cfg(not(feature = "obliv"))]
                let cond = v != -1;

                present.push(cond);
                cond
            })
            .collect::<Vec<_>>();

        let filtered_genotypes = Array1::<Genotype>::from_vec(filtered_genotypes);

        //log!("#sites = {}", filtered_genotypes.len());

        let mcmc_params = {
            let mut sites_bitmask = sites_bitmask.clone();
            sites_bitmask
                .iter_mut()
                .filter(|v| **v)
                .zip(present.iter())
                .for_each(|(b, &p)| *b = p);

            let bps = bps
                .iter()
                .zip(sites_bitmask.iter())
                .filter(|(_, b)| **b)
                .map(|(&cm, _)| cm)
                .collect::<Vec<_>>();

            let mut cms = cms
                .iter()
                .zip(sites_bitmask.iter())
                .filter(|(_, b)| **b)
                .map(|(&cm, _)| cm)
                .collect::<Vec<_>>();
            let first = cms[0];
            cms.iter_mut().for_each(|cm| *cm -= first);

            let (ref_panel_new, afreqs) = common::ref_panel::m3vcf_scan(
                &ref_panel_meta,
                &ref_panel_blocks,
                &sites_bitmask,
                cli.min_m3vcf_unique_haps,
                cli.max_m3vcf_unique_haps,
            );

            //use statrs::statistics::Statistics;
            //let block_n_unique_haps = ref_panel_new
            //.blocks
            //.iter()
            //.map(|b| b.n_unique() as f64)
            //.collect::<Vec<_>>();

            //let block_n_sites = ref_panel_new
            //.blocks
            //.iter()
            //.map(|b| b.n_sites() as f64)
            //.collect::<Vec<_>>();
            //log!("#Blocks: {}", ref_panel_new.blocks.len());

            //log!(
            //"#Block unique haplotypes: {:.3}+/-{:.3}",
            //Statistics::mean(&block_n_unique_haps),
            //Statistics::std_dev(&block_n_unique_haps)
            //);

            //log!(
            //"#Block sites: {:.3}+/-{:.3}",
            //Statistics::mean(&block_n_sites),
            //Statistics::std_dev(&block_n_sites)
            //);

            mcmc::McmcSharedParams::new(
                ref_panel_new,
                bps,
                cms,
                afreqs,
                cli.min_window_len_cm,
                n_pos_window_overlap,
                pbwt_modulo,
                pbwt_depth,
            )
        };

        let phased = mcmc::Mcmc::run(
            &mcmc_params,
            filtered_genotypes.view(),
            &mcmc_iterations,
            rng,
            &sample_id.to_string(),
        );

        #[cfg(feature = "obliv")]
        let phased = phased.map(|v| v.expose());

        let mut phased_with_missing = Array2::zeros((present.len(), 2));

        let mut phased_iter = phased.rows().into_iter();

        for (b, mut r) in present
            .into_iter()
            .zip(phased_with_missing.rows_mut().into_iter())
        {
            if b {
                r.assign(&phased_iter.next().unwrap());
            } else {
                r.fill(-1);
            }
        }
        log!("finished phasing");
        bincode::serialize_into(&mut host_stream, &phased_with_missing).unwrap();
        host_stream.flush().unwrap();
        log!("sent results to host");
        log!("done");
    }
}
