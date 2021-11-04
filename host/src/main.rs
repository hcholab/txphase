#![allow(dead_code)]

mod record;
mod site;

use rust_htslib::bcf;
use std::collections::HashSet;
use std::io::Write;
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::Path;
use std::str::FromStr;

const SP_PORT: u16 = 1234;

fn process_input_bcf(
    input_bcf_path: &Path,
    ref_sites_path: &Path,
) -> (
    Vec<Vec<i8>>,
    Vec<bool>,
    bcf::header::HeaderView,
    Vec<bcf::record::Record>,
) {
    let ref_sites = site::sites_from_csv_path(ref_sites_path).unwrap();

    let (input_sites, input_bcf_header, input_bcf_records) =
        site::sites_from_bcf_path(input_bcf_path).unwrap();

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
            if ref_site == site {
                ref_sites_bitmask.push(true);
                while let Some((target_site, target_record)) = bcf_iter.next() {
                    let target_record = target_record;
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
                assert_eq!(samples[0].len(), ref_sites_bitmask.len());
                break;
            } else {
                ref_sites_bitmask.push(false);
                for sample in samples.iter_mut() {
                    sample.push(-1); // missing
                }
                assert_eq!(samples[0].len(), ref_sites_bitmask.len());
            }
        }
    }

    while ref_sites_iter.next().is_some() {
        // clear
        ref_sites_bitmask.push(false);
        for sample in samples.iter_mut() {
            sample.push(-1); // missing
        }
    }

    //assert_eq!(ref_sites_bitmask.len(), n_sites);
    //assert_eq!(ref_sites_bitmask.iter().filter(|&&b| b).count(), n_overlap);
    //assert_eq!(
    //samples[0].iter().filter(|&&g| g == -1).count(),
    //n_sites - n_overlap
    //);
    //assert_eq!(samples[0].iter().filter(|&&g| g != -1).count(), n_overlap);
    //assert_eq!(input_records_filtered.len(), n_overlap);
    //for sample in &samples {
    //assert_eq!(sample.len(), n_sites);
    //}

    return (
        samples,
        ref_sites_bitmask,
        input_bcf_header,
        input_records_filtered,
    );
}

fn main() {
    let input_bcf_path = "/home/ndokmai/workspace/genome-data/data/giab/son.vcf.gz";
    let ref_sites_path = "/home/ndokmai/workspace/genome-data/data/1kg/old/chr20_sites.csv";

    let (mut target_samples, ref_sites_bitmask, input_bcf_header, input_records_filtered) =
        process_input_bcf(&Path::new(input_bcf_path), &mut Path::new(ref_sites_path));
    let mut target_sample = target_samples.pop().unwrap();

    let (mut ref_panel_meta, ref_panel_block_iter) = m3vcf::load_ref_panel(std::path::Path::new(
        "/home/ndokmai/workspace/genome-data/data/1kg/old/ref_panel.m3vcf.gz",
    ));
    let mut ref_panel_blocks = ref_panel_block_iter.collect::<Vec<_>>();

    assert_eq!(ref_panel_meta.n_markers, target_sample.len());

    {
        let limit = 10000;
        let mut s = 1;
        let mut block_limit = 0;
        for (i, block) in ref_panel_blocks.iter().enumerate() {
            if s + block.nvar - 1 > limit {
                block_limit = i;
                break;
            } else {
                s += block.nvar - 1;
            }
        }
        ref_panel_blocks.drain(block_limit..);
        ref_panel_meta.n_blocks = ref_panel_blocks.len();
        ref_panel_meta.n_markers =
            ref_panel_blocks.iter().map(|b| b.nvar).sum::<usize>() - ref_panel_blocks.len() + 1;
        target_sample.drain(ref_panel_meta.n_markers..);
    }

    eprintln!("Host: n_blocks = {}", ref_panel_meta.n_blocks);
    eprintln!("Host: n_haps = {}", ref_panel_meta.n_haps);
    eprintln!("Host: n_markers = {}", ref_panel_meta.n_markers);

    assert_eq!(ref_panel_meta.n_markers, target_sample.len());

    let mut sp_stream = bufstream::BufStream::new(tcp_keep_connecting(SocketAddr::from((
        IpAddr::from_str("127.0.0.1").unwrap(),
        SP_PORT,
    ))));

    eprintln!("Host: connected to SP");

    bincode::serialize_into(&mut sp_stream, &ref_panel_meta).unwrap();
    bincode::serialize_into(&mut sp_stream, &ref_panel_blocks).unwrap();
    bincode::serialize_into(&mut sp_stream, &target_sample).unwrap();
    sp_stream.flush().unwrap();

    let phased: ndarray::Array2<i8> = bincode::deserialize_from(&mut sp_stream).unwrap();
    let phased = phased
        .columns()
        .into_iter()
        .zip(ref_sites_bitmask.into_iter())
        .filter(|(_, b)| *b)
        .map(|(v, _)| (v[0], v[1]))
        .collect::<Vec<_>>();

    let mut out_vcf = bcf::Writer::from_path(
        &Path::new("phased.vcf.gz"),
        &bcf::header::Header::from_template(&input_bcf_header),
        false,
        bcf::Format::Vcf,
    )
    .unwrap();

    use bcf::record::GenotypeAllele;

    for (genotype, input_record) in phased.into_iter().zip(input_records_filtered) {
        let mut new_record = out_vcf.empty_record();
        new_record.set_rid(input_record.rid());
        new_record.set_pos(input_record.pos());
        new_record.set_id(&input_record.id()).unwrap();
        new_record.set_alleles(&input_record.alleles()).unwrap();
        new_record
            .push_genotypes(&[
                GenotypeAllele::Phased(genotype.0 as i32),
                GenotypeAllele::Phased(genotype.1 as i32),
            ])
            .unwrap();
        out_vcf.write(&new_record).unwrap();
    }

    eprintln!("Host: done");
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
