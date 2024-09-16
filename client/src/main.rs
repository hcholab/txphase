use clap::{value_parser, Parser};
use m3vcf::Site;
use ndarray::Array2;
use rust_htslib::bcf;
use std::collections::HashSet;
use std::io::Write;
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::str::FromStr;

mod record;
mod site;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "127.0.0.1")]
    sp_ip_address: String,
    #[arg(long, default_value_t = 7777)]
    sp_port: u16,
    #[arg(long, default_value_t = 7776)]
    host_port: u16,
    #[arg(long, value_parser=value_parser!(PathBuf))]
    input: PathBuf,
    #[arg(long, value_parser=value_parser!(PathBuf), default_value="phased.vcf.gz")]
    output: PathBuf,
}

macro_rules! log {
    ($($arg:tt)*) => {
        println!("Client:: {}", format!($($arg)*));
    };
}

fn main() {
    let cli = Cli::parse();
    log!("SP IP address:\t\t{}", cli.sp_ip_address);
    log!("SP host port:\t\t{}", cli.host_port);
    log!("Input:\t\t\t{:?}", cli.input);
    log!("Output:\t\t{:?}", cli.output);

    let sp_ip_address = IpAddr::from_str(&cli.sp_ip_address).unwrap();
    let host_socket = SocketAddr::from((sp_ip_address, cli.host_port));

    log!("connecting to SP @ {} ...", host_socket);

    let mut host_stream = bufstream::BufStream::new(tcp_keep_connecting(host_socket));

    log!("connected to SP");

    let sites: Vec<Site> = bincode::deserialize_from(&mut host_stream).unwrap();

    log!("received site information from host");

    let (target_samples, afreq_bitmask, input_bcf_header, input_records_filtered) =
        process_input(&cli.input, &sites);

    bincode::serialize_into(&mut host_stream, &afreq_bitmask).unwrap();
    //TODO: encrypt this with SP's secret key
    bincode::serialize_into(&mut host_stream, &target_samples).unwrap();

    host_stream.flush().unwrap();

    log!("sent input to host");

    let all_results: Vec<Array2<i8>> = bincode::deserialize_from(&mut host_stream).unwrap();

    log!("sent output from host");

    write_vcf(
        &cli.output,
        &all_results[..],
        &input_bcf_header,
        &input_records_filtered,
    );

    log!("output written to VCF file");
    log!("done");
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
