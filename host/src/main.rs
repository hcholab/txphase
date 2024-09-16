#![allow(dead_code)]

mod geneticmap;
mod record;
mod site;

use ndarray::Array2;
use std::io::Write;
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::str::FromStr;

use clap::{value_parser, Parser};

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value_t = 7776)]
    client_port: u16,
    #[arg(long, default_value_t = 7777)]
    worker_port_base: u16,
    #[arg(long, default_value_t = 1)]
    n_workers: usize,
    #[arg(long)]
    filter_maf: Option<f32>,
    #[arg(long, value_parser=value_parser!(PathBuf))]
    ref_panel: PathBuf,
    #[arg(long, value_parser=value_parser!(PathBuf))]
    genetic_map: PathBuf,
}

macro_rules! log {
    ($($arg:tt)*) => {
        println!("Host:: {}", format!($($arg)*));
    };
}

fn main() {
    let cli = Cli::parse();
    log!("Client port : \t\t{}", cli.client_port);
    log!("Worker port base: \t{}", cli.worker_port_base);
    log!("# workers: \t\t{}", cli.n_workers);
    log!("Reference panel: \t{:?}", cli.ref_panel);
    log!("Genetic map: \t\t{:?}", cli.genetic_map);

    let (mut client_stream, client_socket) = TcpListener::bind(SocketAddr::from((
        IpAddr::from_str("127.0.0.1").unwrap(),
        cli.client_port,
    )))
    .unwrap()
    .accept()
    .unwrap();

    log!("Client connected from {}", client_socket);

    let sites = m3vcf::read_sites(&cli.ref_panel);

    bincode::serialize_into(&mut client_stream, &sites).unwrap();
    log!("site information sent to Client");

    let afreq_bitmask: Vec<bool> = bincode::deserialize_from(&mut client_stream).unwrap();
    let target_samples: Vec<Vec<i8>> = bincode::deserialize_from(&mut client_stream).unwrap();

    let (ref_panel_meta, ref_panel_block_iter) = m3vcf::load_ref_panel(&cli.ref_panel);
    let ref_panel_blocks = ref_panel_block_iter.collect::<Vec<_>>();

    let genetic_map = geneticmap::genetic_map_from_csv_path(&cli.genetic_map).unwrap();
    let bps = sites.iter().map(|s| s.pos).collect::<Vec<_>>();
    let interpolated_cms = geneticmap::interpolate_cm(&genetic_map, &sites);

    let mut afreqs = Vec::new();

    for (i, block) in ref_panel_blocks.iter().enumerate() {
        if i != ref_panel_blocks.len() - 1 {
            afreqs.extend(block.afreq.iter().take(block.nvar - 1).cloned());
        } else {
            afreqs.extend(block.afreq.iter().cloned());
        }
    }

    const THRES: f32 = 0.001;

    let (afreqs_filter, sites) = if let Some(&maf_threshold) = cli.filter_maf.as_ref() {
        let afreqs_filter = afreqs
            .iter()
            .map(|f| f.min(1. - f) > maf_threshold)
            .collect::<Vec<_>>();

        let sites = afreqs_filter
            .iter()
            .zip(sites.iter())
            .filter_map(|(b, s)| if *b { Some(s.clone()) } else { None })
            .collect::<Vec<_>>();

        (afreqs_filter, sites)
    } else {
        (vec![true; afreqs.len()], sites)
    };

    let mut ref_sites_bitmask = vec![false; sites.len()];

    afreqs_filter
        .iter()
        .zip(ref_sites_bitmask.iter_mut())
        .filter_map(|(a, b)| if *a { Some(b) } else { None })
        .zip(afreq_bitmask.iter())
        .for_each(|(b, a)| *b |= *a);

    let sample_id = std::sync::atomic::AtomicUsize::new(0);
    let worker_id = std::sync::atomic::AtomicUsize::new(0);
    let mut all_results: Vec<Option<Array2<i8>>> = vec![None; target_samples.len()];
    std::thread::scope(|s| {
        let join_handles = (0..cli.n_workers)
            .map(|_| {
                s.spawn(|| {
                    let worker_id = worker_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let socket = SocketAddr::from((
                        IpAddr::from_str("127.0.0.1").unwrap(),
                        cli.worker_port_base + worker_id as u16,
                    ));
                    log!("connecting to Worker {worker_id} @ {} ...", socket);
                    let mut worker_stream = bufstream::BufStream::new(tcp_keep_connecting(socket));
                    log!("connected to Worker {worker_id}");
                    bincode::serialize_into(&mut worker_stream, &(worker_id as u16)).unwrap();
                    bincode::serialize_into(&mut worker_stream, &ref_panel_meta).unwrap();
                    bincode::serialize_into(&mut worker_stream, &ref_panel_blocks).unwrap();
                    bincode::serialize_into(&mut worker_stream, &ref_sites_bitmask).unwrap();
                    bincode::serialize_into(&mut worker_stream, &interpolated_cms).unwrap();
                    bincode::serialize_into(&mut worker_stream, &bps).unwrap();
                    log!("reference panel sent to Worker {worker_id}");
                    worker_stream.flush().unwrap();
                    let mut results = Vec::new();

                    loop {
                        let sample_id = sample_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if sample_id >= target_samples.len() {
                            break;
                        }
                        bincode::serialize_into(&mut worker_stream, &target_samples[sample_id])
                            .unwrap();
                        bincode::serialize_into(&mut worker_stream, &sample_id).unwrap();
                        worker_stream.flush().unwrap();
                        log!("Sample {sample_id} sent to Worker {worker_id}");
                        results.push((
                            sample_id,
                            bincode::deserialize_from::<_, Array2<i8>>(&mut worker_stream).unwrap(),
                        ));
                        log!("results received from Worker {worker_id}");
                    }
                    results
                })
            })
            .collect::<Vec<_>>();

        for join_hanle in join_handles {
            let results = join_hanle.join().unwrap();
            for (sample_id, phased) in results {
                all_results[sample_id] = Some(phased);
            }
        }
    });

    let all_results = all_results
        .into_iter()
        .map(|v| v.unwrap())
        .collect::<Vec<_>>();

    log!("received all results");

    bincode::serialize_into(&mut client_stream, &all_results).unwrap();

    log!("results sent to Client");
    log!("done");
}

fn tcp_keep_connecting(addr: SocketAddr) -> TcpStream {
    let stream;
    loop {
        if let Ok(s) = TcpStream::connect(addr) {
            stream = Some(s);
            break;
        };
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    stream.unwrap()
}
