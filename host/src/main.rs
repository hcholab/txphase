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
use std::path::Path;
use std::str::FromStr;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 6);
    let sp_port = args[1].parse::<u16>().unwrap();
    let ref_panel_path = &args[2];
    let genetic_map_path = &args[3];
    let input_path = &args[4];
    let output_path = &args[5];

    eprintln!("Port: \t\t\t{sp_port}");
    eprintln!("Reference panel: \t{ref_panel_path}");
    eprintln!("Genetic map: \t\t{genetic_map_path}");
    eprintln!("Input: \t\t\t{input_path}");
    eprintln!("Output: \t\t{output_path}");

    let sites = m3vcf::read_sites(std::path::Path::new(ref_panel_path));

    let (ref_panel_meta, ref_panel_block_iter) =
        m3vcf::load_ref_panel(std::path::Path::new(ref_panel_path));
    let ref_panel_blocks = ref_panel_block_iter.collect::<Vec<_>>();

    let genetic_map = geneticmap::genetic_map_from_csv_path(&Path::new(genetic_map_path)).unwrap();
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

    let afreq_filter = afreqs
        .iter()
        .map(|f| f.min(1. - f) > THRES)
        .collect::<Vec<_>>();

    println!("count {}", afreq_filter.iter().filter(|&&b| b).count());

    let afreq_sites = afreq_filter
        .iter()
        .zip(sites.iter())
        .filter_map(|(b, s)| if *b { Some(s.clone()) } else { None })
        .collect::<Vec<_>>();

    let (target_samples, afreq_bitmask, input_bcf_header, input_records_filtered) =
        process_input(&Path::new(input_path), &afreq_sites);

    let mut ref_sites_bitmask = vec![false; sites.len()];

    afreq_filter
        .iter()
        .zip(ref_sites_bitmask.iter_mut())
        .filter_map(|(a, b)| if *a { Some(b) } else { None })
        .zip(afreq_bitmask.iter())
        .for_each(|(b, a)| *b |= *a);

    let mut sp_stream = bufstream::BufStream::new(tcp_keep_connecting(SocketAddr::from((
        IpAddr::from_str("127.0.0.1").unwrap(),
        sp_port,
    ))));

    eprintln!("Host: connected to SP");

    bincode::serialize_into(&mut sp_stream, &ref_panel_meta).unwrap();
    bincode::serialize_into(&mut sp_stream, &ref_panel_blocks).unwrap();
    bincode::serialize_into(&mut sp_stream, &ref_sites_bitmask).unwrap();
    bincode::serialize_into(&mut sp_stream, &interpolated_cms).unwrap();
    bincode::serialize_into(&mut sp_stream, &bps).unwrap();
    bincode::serialize_into(&mut sp_stream, &target_samples).unwrap();
    sp_stream.flush().unwrap();

    let phased: Vec<ndarray::Array2<i8>> = bincode::deserialize_from(&mut sp_stream).unwrap();

    println!("phased len = {}", phased.len());

    write_vcf(
        &output_path,
        &phased[..],
        &input_bcf_header,
        &input_records_filtered,
    );

    eprintln!("Host: done");
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
    //let ref_sites = site::sites_from_csv_path(ref_sites_path).unwrap();

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
    file_name: &str,
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
        &Path::new(file_name),
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
