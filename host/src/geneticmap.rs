use crate::site::Site;
use std::path::Path;

pub fn genetic_map_from_csv_path(path: &Path) -> anyhow::Result<Vec<(u32, f64)>> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(path)?;
    let mut out = reader
        .records()
        .map(|line| {
            let line = line.unwrap();
            (
                line.get(0).unwrap().parse().unwrap(),
                line.get(2).unwrap().parse().unwrap(),
            )
        })
        .collect::<Vec<(u32, f64)>>();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

pub fn interpolate_cm(genetic_map: &[(u32, f64)], sites: &[Site]) -> Vec<f64> {
    let mut interpolated = Vec::<f64>::with_capacity(sites.len());

    let first_map = genetic_map.first().unwrap();
    let last_map = genetic_map.last().unwrap();
    let mean_rate = (last_map.1 - first_map.1) / (last_map.0 - first_map.0) as f64;

    let mut sites_iter = sites.iter().enumerate();

    let mut site_ptr = sites_iter.next();

    // Head positions
    while let Some((site_i, site)) = site_ptr {
        if site.pos >= first_map.0 {
            if site_i > 0 {
                interpolate_cm_head(&first_map, mean_rate, &sites[..site_i], &mut interpolated);
            }
            break;
        }
        site_ptr = sites_iter.next();
    }

    // Middle positions
    for (start_map, end_map) in genetic_map.iter().zip(genetic_map.iter().skip(1)) {
        if let Some((_site_i, site)) = site_ptr {
            if site.pos == start_map.0 {
                interpolated.push(start_map.1);
                site_ptr = sites_iter.next();
            }
        } else {
            break;
        }

        if let Some((site_i, site)) = site_ptr {
            if site.pos > start_map.0 && site.pos < end_map.0 {
                let start_site_i = site_i;

                site_ptr = sites_iter.next();
                while let Some((site_i, site)) = site_ptr {
                    if site.pos >= end_map.0 {
                        interpolate_cm_middle(
                            &start_map,
                            &end_map,
                            &sites[start_site_i..site_i],
                            &mut interpolated,
                        );
                        break;
                    }
                    site_ptr = sites_iter.next();
                }
            }
        } else {
            break;
        }

        if let Some((_site_i, site)) = site_ptr {
            if site.pos == end_map.0 {
                interpolated.push(end_map.1);
                site_ptr = sites_iter.next();
            }
        } else {
            break;
        }
    }

    //Tail positions
    if let Some((site_i, _)) = site_ptr {
        interpolate_cm_tail(&last_map, mean_rate, &sites[site_i..], &mut interpolated);
    }

    // Shift by first cM
    let baseline = interpolated[0];
    interpolated.iter_mut().for_each(|v| *v -= baseline);

    assert_eq!(interpolated.len(), sites.len());
    interpolated
}

fn interpolate_cm_middle(
    first_map: &(u32, f64),
    last_map: &(u32, f64),
    sites: &[Site],
    out: &mut Vec<f64>,
) {
    let rate = (last_map.1 - first_map.1) / (last_map.0 - first_map.0) as f64;
    interpolate_cm_tail(first_map, rate, sites, out);
}

fn interpolate_cm_head(first_map: &(u32, f64), mean_rate: f64, sites: &[Site], out: &mut Vec<f64>) {
    sites.iter().for_each(|s| {
        let dist = first_map.0 - s.pos;
        let cm = first_map.1 - mean_rate * dist as f64;
        out.push(cm);
    });
}

fn interpolate_cm_tail(last_map: &(u32, f64), mean_rate: f64, sites: &[Site], out: &mut Vec<f64>) {
    sites.iter().for_each(|s| {
        let dist = s.pos - last_map.0;
        let cm = last_map.1 + mean_rate * dist as f64;
        out.push(cm);
    });
}
