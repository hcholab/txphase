#[derive(Debug)]
struct Record {
    pub chr: usize,
    pub round: usize,
    pub ks: Vec<(f32, f32)>,
}

const N_STDEV: f32 = 1.;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let file = std::fs::File::open(&args[1]).unwrap();
    let mut rdr = csv::Reader::from_reader(file);
    let mut records = Vec::new();
    for result in rdr.records() {
        let mut record = result.unwrap();
        record.trim();
        let mut record_iter = record.iter();
        let chr = record_iter.next().unwrap().parse::<usize>().unwrap();
        let round = record_iter.next().unwrap().parse::<usize>().unwrap();
        let mut ks = Vec::new();
        for k in record_iter {
            if k.is_empty() {
                break;
            }
            let k = k.split("+/-").collect::<Vec<_>>();
            ks.push((k[0].parse::<f32>().unwrap(), k[1].parse::<f32>().unwrap()));
        }

        records.push(Record { chr, round, ks });
    }

    records.sort_by_key(|k| (k.chr, k.round));

    let mut chr = 0;
    let mut all_stats = Vec::new();
    let mut new_stats = Vec::new();
    for record in records {
        if record.chr != chr {
            chr = record.chr;
            if !new_stats.is_empty() {
                all_stats.push(new_stats);
            }
            new_stats = record.ks.into_iter().map(|(avg, stdev)| avg + N_STDEV*stdev).collect();
        } else {
            new_stats.iter_mut().zip(record.ks.into_iter()).for_each(|(cur, (avg, stdev))| {
                *cur = cur.max(avg + N_STDEV*stdev);
            });
        }
    } 
    all_stats.push(new_stats);
    let mut all_max = Vec::new();
    println!("\t\t b1\t b2\t b3\t b4\t b5\t p1\t b6\t p2\t b7\t p2\t m1\t m2\t m3\t m4\t m5");
    for (i, s) in all_stats.into_iter().enumerate() {
        let rounded = s.into_iter().map(|v| v.ceil() as usize).collect::<Vec<_>>();
        let i = i + 1;
        print!("chr {i}:\t\t");
        for v in &rounded {
            print!("{v}\t");
        }
        println!("");
        if i==1 {
            all_max = rounded;
        } else {
            all_max.iter_mut().zip(rounded.into_iter()).for_each(|(a, b)| *a = (*a).max(b));
        }
    }
    print!("all:\t\t");
    for v in &all_max{
        print!("{v}\t");
    }
    println!("");
}
