#![allow(dead_code)]

mod geneticmap;
mod record;
mod site;

use m3vcf::Site;
use ndarray::Array2;
use rust_htslib::bcf;
use std::collections::HashSet;
use std::io::Write;
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::{value_parser, Parser};

#[derive(Parser, Debug)]
struct Cli {
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
    #[arg(long, value_parser=value_parser!(PathBuf))]
    input: PathBuf,
    #[arg(long, value_parser=value_parser!(PathBuf), default_value="phased.vcf.gz")]
    output: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    println!("Worker port base: \t\t\t{}", cli.worker_port_base);
    println!("Reference panel: \t{:?}", cli.ref_panel);
    println!("Genetic map: \t\t{:?}", cli.genetic_map);
    println!("Input: \t\t\t{:?}", cli.input);
    println!("Output: \t\t{:?}", cli.output);

    let sites = m3vcf::read_sites(&cli.ref_panel);

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

    let (target_samples, afreq_bitmask, input_bcf_header, input_records_filtered) =
        process_input(&cli.input, &sites);

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
                    let mut worker_stream =
                        bufstream::BufStream::new(tcp_keep_connecting(SocketAddr::from((
                            IpAddr::from_str("127.0.0.1").unwrap(),
                            cli.worker_port_base + worker_id as u16,
                        ))));
                    println!("Host: connected to Worker {worker_id}");
                    bincode::serialize_into(&mut worker_stream, &ref_panel_meta).unwrap();
                    bincode::serialize_into(&mut worker_stream, &ref_panel_blocks).unwrap();
                    bincode::serialize_into(&mut worker_stream, &ref_sites_bitmask).unwrap();
                    bincode::serialize_into(&mut worker_stream, &interpolated_cms).unwrap();
                    bincode::serialize_into(&mut worker_stream, &bps).unwrap();
                    worker_stream.flush().unwrap();
                    let mut results = Vec::new();

                    loop {
                        let sample_id = sample_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if sample_id >= target_samples.len() {
                            break;
                        }
                        println!("Host: sending Sample {sample_id} to Worker {worker_id}");
                        bincode::serialize_into(&mut worker_stream, &target_samples[sample_id])
                            .unwrap();
                        bincode::serialize_into(&mut worker_stream, &sample_id).unwrap();
                        worker_stream.flush().unwrap();
                        results.push((
                            sample_id,
                            bincode::deserialize_from::<_, Array2<i8>>(&mut worker_stream).unwrap(),
                        ));
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

    write_vcf(
        &cli.output,
        &all_results[..],
        &input_bcf_header,
        &input_records_filtered,
    );

    println!("Host: done");
}

fn process_input(
    input_path: &Path,
    ref_sites: &[Site],
) -> (
    Vec<Vec<i8>>,
    Vec<bool>,
    bcf::header::HeaderView,
    Vec<bcf::record::Record>,
) {
    let (input_sites, input_bcf_header, input_bcf_records) =
        site::sites_from_bcf_path(input_path).unwrap();

    assert_eq!(input_sites.len(), input_bcf_records.len());

    let mut overlap = {
        let ref_sites_set = ref_sites.iter().collect::<HashSet<_>>();
        let input_sites_set = input_sites.iter().collect::<HashSet<_>>();
        ref_sites_set
            .intersection(&input_sites_set)
            .cloned()
            .cloned()
            .collect::<Vec<_>>()
    };

    overlap.sort_unstable();

    let n_overlap = overlap.len();
    let n_sites = ref_sites.len();

    let mut bcf_iter = input_sites.into_iter().zip(input_bcf_records.into_iter());
    let mut ref_sites_iter = ref_sites.into_iter();
    let mut ref_sites_bitmask = Vec::with_capacity(n_sites);

    let n_samples = input_bcf_header.sample_count() as usize;
    let mut samples: Vec<Vec<i8>> = vec![Vec::with_capacity(n_sites); n_samples];
    let mut input_records_filtered = Vec::with_capacity(n_overlap);

    for site in overlap {
        while let Some(ref_site) = ref_sites_iter.next() {
            if ref_site == &site {
                ref_sites_bitmask.push(true);
                break;
            } else {
                ref_sites_bitmask.push(false);
            }
        }

        while let Some((target_site, target_record)) = bcf_iter.next() {
            if target_site == site {
                let genotypes = target_record.genotypes().unwrap();
                for (i, sample) in samples.iter_mut().enumerate() {
                    let genotype: record::Genotype = genotypes.get(i).into();
                    let genotype: i8 = genotype.as_unphased().into();
                    sample.push(genotype);
                }
                input_records_filtered.push(target_record);
                break;
            }
        }
    }

    while ref_sites_iter.next().is_some() {
        ref_sites_bitmask.push(false);
    }

    return (
        samples,
        ref_sites_bitmask,
        input_bcf_header,
        input_records_filtered,
    );
}

fn write_vcf(
    file_name: &std::path::Path,
    phased: &[Array2<i8>],
    input_bcf_header: &bcf::header::HeaderView,
    input_records_filtered: &[bcf::record::Record],
) {
    use bcf::record::GenotypeAllele;

    let mut genotypes = vec![Vec::new(); input_records_filtered.len()];

    for sample in phased {
        for (r, g) in sample.rows().into_iter().zip(genotypes.iter_mut()) {
            if r[0] != -1 && r[1] != -1 {
                g.push(GenotypeAllele::Phased(r[0] as i32));
                g.push(GenotypeAllele::Phased(r[1] as i32));
            } else {
                g.push(GenotypeAllele::PhasedMissing);
                g.push(GenotypeAllele::PhasedMissing);
            }
        }
    }

    let mut out_vcf = bcf::Writer::from_path(
        file_name,
        &bcf::header::Header::from_template(&input_bcf_header),
        false,
        bcf::Format::Vcf,
    )
    .unwrap();

    for (genotype, input_record) in genotypes.into_iter().zip(input_records_filtered) {
        let mut new_record = out_vcf.empty_record();
        new_record.set_rid(input_record.rid());
        new_record.set_pos(input_record.pos());
        new_record.set_id(&input_record.id()).unwrap();
        new_record.set_alleles(&input_record.alleles()).unwrap();
        new_record.push_genotypes(&genotype).unwrap();
        out_vcf.write(&new_record).unwrap();
    }
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
