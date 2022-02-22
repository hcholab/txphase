#![allow(dead_code)]

mod geneticmap;
mod record;
mod site;

use ndarray::ArrayView2;
use rust_htslib::bcf;
use std::collections::HashSet;
use std::io::Write;
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::Path;
use std::str::FromStr;

const SP_PORT: u16 = 1234;
const CHR: usize = 20;

fn main() {
    let genetic_map_path = &format!("/home/ndokmai/workspace/shapeit4/maps/chr{}.b37.gmap", CHR);
    let input_bcf_path = &format!("/home/ndokmai/workspace/genome-data/data/giab/Ch37_chr{}/son.vcf.gz", CHR);
    let ref_panel_path = &format!("/home/ndokmai/workspace/genome-data/data/1kg/m3vcf/{}.1000g.Phase3.v5.With.Parameter.Estimates.m3vcf.gz", CHR);
    let ref_sites_path = &format!("/home/ndokmai/workspace/genome-data/data/1kg/sites/chr{}_sites.csv", CHR);

    let genetic_map = geneticmap::genetic_map_from_csv_path(&Path::new(genetic_map_path)).unwrap();
    let sites = site::sites_from_csv_path(&Path::new(ref_sites_path)).unwrap();
    let interpolated_cms = geneticmap::interpolate_cm(&genetic_map, &sites);

    let (ref_panel_meta, ref_panel_block_iter) =
        m3vcf::load_ref_panel(std::path::Path::new(ref_panel_path));
    let ref_panel_blocks = ref_panel_block_iter.collect::<Vec<_>>();

    let (mut target_samples, ref_sites_bitmask, input_bcf_header, input_records_filtered) =
        process_input_bcf(&Path::new(input_bcf_path), &mut Path::new(ref_sites_path));
    let target_sample = target_samples.pop().unwrap();

    let mut sp_stream = bufstream::BufStream::new(tcp_keep_connecting(SocketAddr::from((
        IpAddr::from_str("127.0.0.1").unwrap(),
        SP_PORT,
    ))));

    eprintln!("Host: connected to SP");

    bincode::serialize_into(&mut sp_stream, &ref_panel_meta).unwrap();
    bincode::serialize_into(&mut sp_stream, &ref_panel_blocks).unwrap();
    bincode::serialize_into(&mut sp_stream, &ref_sites_bitmask).unwrap();
    bincode::serialize_into(&mut sp_stream, &interpolated_cms).unwrap();
    bincode::serialize_into(&mut sp_stream, &target_sample).unwrap();
    sp_stream.flush().unwrap();

    let phased: ndarray::Array2<i8> = bincode::deserialize_from(&mut sp_stream).unwrap();
    write_vcf(
        "phased.vcf.gz",
        phased.view(),
        &input_bcf_header,
        &input_records_filtered,
    );

    eprintln!("Host: done");
}

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
                break;
            } else {
                ref_sites_bitmask.push(false);
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
    phased: ArrayView2<i8>,
    input_bcf_header: &bcf::header::HeaderView,
    input_records_filtered: &[bcf::record::Record],
) {
    use bcf::record::GenotypeAllele;
    let phased = phased
        .rows()
        .into_iter()
        .map(|v| (v[0], v[1]))
        .collect::<Vec<_>>();

    let mut out_vcf = bcf::Writer::from_path(
        &Path::new(file_name),
        &bcf::header::Header::from_template(&input_bcf_header),
        false,
        bcf::Format::Vcf,
    )
    .unwrap();

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
