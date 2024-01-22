#![feature(stmt_expr_attributes)]
#![feature(iter_array_chunks)]
#![allow(dead_code)]

mod genotype_graph;
mod hmm;
mod mcmc;
mod rss_hmm;
mod utils;
mod variants;

#[cfg(not(feature = "obliv"))]
mod neighbors_finding;
#[cfg(not(feature = "obliv"))]
mod pbwt;

#[cfg(feature = "obliv")]
mod dynamic_fixed;

#[cfg(feature = "obliv")]
mod inner {
    pub use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpI32, TpI8, TpU32, TpU64, TpU8};
    pub type Genotype = TpI8;
    pub type UInt = TpU32;
    pub type Usize = TpU64;
    pub type Int = TpI32;
    pub type U8 = TpU8;
    pub type Bool = TpBool;
    pub const F: usize = 52;
    pub type Real = tp_fixedpoint::TpFixed64<F>;
}

#[cfg(not(feature = "obliv"))]
mod inner {
    pub type Genotype = i8;
    pub type UInt = u32;
    pub type Usize = usize;
    pub type Int = i32;
    pub type U8 = u8;
    pub type Bool = bool;
    pub type Real = f64;
}

use inner::*;

use crate::mcmc::IterOption;
use ndarray::{Array1, Array2};
use rand::{RngCore, SeedableRng};
use rayon::prelude::*;
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::str::FromStr;

const N_THREADS: usize = 40;

pub fn log_template(str1: &str, str2: &str) -> String {
    format!("\t* {str1}\t: {str2}")
}

//use std::cell::RefCell;
//use std::fs::File;
//use std::io::BufWriter;
//thread_local! {
//pub static DEBUG_FILE: RefCell<BufWriter<File>> = RefCell::new(BufWriter::new(File::create("debug.txt").unwrap()));
//}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 2);
    let host_port = args[1].parse::<u16>().unwrap();

    let min_window_len_cm = 4.0;
    let min_het_rate = 0.1f64;
    let n_pos_window_overlap = (3. / min_het_rate).ceil() as usize;

    println!("Parameters:");

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

    let cms: Vec<f64> = bincode::deserialize_from(&mut host_stream).unwrap();
    let bps: Vec<u32> = bincode::deserialize_from(&mut host_stream).unwrap();
    let genotypes: Vec<Vec<i8>> = bincode::deserialize_from(&mut host_stream).unwrap();
    let iterations = [
        IterOption::Burnin(5),
        IterOption::Pruning(1),
        IterOption::Burnin(1),
        IterOption::Pruning(1),
        IterOption::Burnin(1),
        IterOption::Pruning(1),
        IterOption::Main(5),
        //IterOption::Main(1),
    ];

    let pbwt_depth = ((9. - ((ref_panel_meta.n_haps / 2) as f64).log10()).round() as usize)
        .min(8)
        .max(2);

    let pbwt_modulo = ((((ref_panel_meta.n_haps / 2) as f64).ln() - 50f64.ln() + 1.) * 0.01)
        .min(0.15)
        .max(0.005);

    println!("pbwt-depth = {pbwt_depth}");
    println!("pbwt-modulo = {:.3}", pbwt_modulo);

    let phase_single = false;

    if phase_single {
        let seed = rand::thread_rng().next_u64();
        println!("seed: {seed}");
        let rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        #[cfg(feature = "obliv")]
        let genotypes = genotypes[0]
            .iter()
            .map(|&v| Genotype::protect(v))
            .collect::<Vec<_>>();

        #[cfg(not(feature = "obliv"))]
        let genotypes = genotypes[0].clone();

        let mut present = Vec::new();
        let filtered_genotypes = genotypes
            .iter()
            .filter(|&v| {
                #[cfg(feature = "obliv")]
                let cond = v.tp_not_eq(&-1).expose();

                #[cfg(not(feature = "obliv"))]
                let cond = *v != -1;

                present.push(cond);
                cond
            })
            .cloned()
            .collect::<Vec<_>>();

        let filtered_genotypes = Array1::<Genotype>::from_vec(filtered_genotypes);

        println!("#sites = {}", filtered_genotypes.len());

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

            //{
            //use std::io::Write;
            //DEBUG_FILE.with(|f| {
            //let mut f = f.borrow_mut();
            //writeln!(*f, "## bit positions and genetic map ##").unwrap();
            //for (b, c) in bps.iter().zip(cms.iter()) {
            //writeln!(&mut *f, "{b}: {:.4}",c).unwrap();
            //}
            //});
            //}

            let (ref_panel_new, afreqs) =
                common::ref_panel::m3vcf_scan(&ref_panel_meta, &ref_panel_blocks, &sites_bitmask);

            mcmc::McmcSharedParams::new(
                ref_panel_new,
                bps,
                cms,
                afreqs,
                min_window_len_cm,
                n_pos_window_overlap,
                pbwt_modulo,
                pbwt_depth,
            )
        };

        drop(ref_panel_meta);
        drop(ref_panel_blocks);
        drop(bps);
        drop(cms);
        drop(sites_bitmask);

        let phased = mcmc::Mcmc::run(
            &mcmc_params,
            filtered_genotypes.view(),
            &iterations,
            rng,
            &1.to_string(),
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
        bincode::serialize_into(&mut host_stream, &vec![phased_with_missing]).unwrap();
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(N_THREADS)
            .build_global()
            .unwrap();

        let counter = std::sync::atomic::AtomicUsize::new(1);

        let all_phased = genotypes
            .into_par_iter()
            .map(|genotypes| {
                let id = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                //let seed = rand::thread_rng().next_u64();
                let seed = 12727004758508603152;
                println!("{}", &log_template("Seed", &format!("{seed}")));

                let rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

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

                println!("#sites = {}", filtered_genotypes.len());

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
                    );

                    let block_sizes = ref_panel_new
                        .blocks
                        .iter()
                        .map(|b| b.n_unique() as f64)
                        .collect::<Vec<_>>();
                    use statrs::statistics::Statistics;
                    println!(
                        "Block sizes: {:.3}+/-{:.3}",
                        Statistics::mean(&block_sizes),
                        Statistics::std_dev(&block_sizes)
                    );

                    mcmc::McmcSharedParams::new(
                        ref_panel_new,
                        bps,
                        cms,
                        afreqs,
                        min_window_len_cm,
                        n_pos_window_overlap,
                        pbwt_modulo,
                        pbwt_depth,
                    )
                };

                let phased = mcmc::Mcmc::run(
                    &mcmc_params,
                    filtered_genotypes.view(),
                    &iterations,
                    rng,
                    &id.to_string(),
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
                phased_with_missing
            })
            .collect::<Vec<_>>();

        //println!("Insert: {} ms", compressed_pbwt::nn::INSERT_T.lock().unwrap().as_millis());
        //println!("Init: {} ms", compressed_pbwt::nn::INIT_T.lock().unwrap().as_millis());
        //println!("Neighbors: {} ms", compressed_pbwt::nn::NEIGH_T.lock().unwrap().as_millis());
        //println!("Update: {} ms", compressed_pbwt::nn::UPDATE_T.lock().unwrap().as_millis());

        bincode::serialize_into(&mut host_stream, &all_phased).unwrap();
    }
}
