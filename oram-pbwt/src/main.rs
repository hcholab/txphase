use oram_sgx::*;
use rand::Rng;
use std::time::Instant;

fn print_marker(shift: usize) {
    for _i in 0..shift {
        print!(" ");
    }
    print!("+\n");
}

fn gen_ref_panel(n_samples: usize, n_markers: usize, mut rng: impl Rng) -> Vec<Vec<u8>> {
    (0..n_markers)
        .into_iter()
        .map(|_| {
            (0..n_samples)
                .into_iter()
                .map(|_| rng.gen_range(0..2))
                .collect()
        })
        .collect()
}

fn gen_target_sample(n_markers: usize, mut rng: impl Rng) -> Vec<u8> {
    (0..n_markers)
        .into_iter()
        .map(|_| rng.gen_range(0..2))
        .collect()
}

fn main() {
    let n = 100000; // # samples
    let m = 10; // # markers
    let s = 4;

    // Random generation
    let mut rng = rand::thread_rng();
    let x = gen_ref_panel(n, m, &mut rng);
    let t = gen_target_sample(m, &mut rng);

    let mut now = Instant::now();

    /* PBWT construction */
    let mut a_mat = vec![vec![0; n]; m]; // positional prefix arrays
    let mut d_mat = vec![vec![0; n]; m]; // divergence arrays

    // Extra cache for querying
    let mut u_mat = vec![vec![0; n]; m];
    let mut p_mat = vec![vec![0; n]; m];
    let mut q_mat = vec![vec![0; n]; m];
    let mut pr_mat = vec![vec![0; n]; m];
    let mut qr_mat = vec![vec![0; n]; m];

    let mut a = vec![0; n];
    let mut b = vec![0; n];
    let mut d = vec![0; n];
    let mut e = vec![0; n];
    for i in 0..m {
        let mut u = 0;
        let mut v = 0;
        let mut p = i;
        let mut q = i;
        for j in 0..n {
            let dki = if i == 0 { 0 } else { d_mat[i - 1][j] };
            let aki = if i == 0 { j } else { a_mat[i - 1][j] };
            if dki > p {
                p = dki;
            }
            if dki > q {
                q = dki;
            }
            if x[i][aki] == 0 {
                a[u] = aki;
                d[u] = p;
                u = u + 1;
                p = 0;
            } else {
                b[v] = aki;
                e[v] = q;
                v = v + 1;
                q = 0;
            }
            u_mat[i][j] = u;
            p_mat[i][j] = p;
            q_mat[i][j] = q;
        }
        for j in 0..u {
            a_mat[i][j] = a[j];
            d_mat[i][j] = d[j];
        }
        for j in u..n {
            a_mat[i][j] = b[j - u];
            d_mat[i][j] = e[j - u];
        }

        // Reverse pass (for caching pr and qr)
        p = i;
        q = i;
        for j in (0..n).rev() {
            let dki = if i == 0 || j == n - 1 {
                0
            } else {
                d_mat[i - 1][j + 1] as usize
            };
            let aki = if i == 0 { j } else { a_mat[i - 1][j] as usize };

            if dki > p {
                p = dki;
            }
            if dki > q {
                q = dki;
            }
            if x[i][aki] == 0 {
                p = 0;
            } else {
                q = 0;
            }

            pr_mat[i][j] = p;
            qr_mat[i][j] = q;
        }
    }

    println!(
        "PBWT construction in {} ms",
        (Instant::now() - now).as_millis()
    );
    now = Instant::now();

    /* ORAM construction (performed in enclave) */
    let mut pbwt_oram = Vec::with_capacity(m);
    const VALUE_SIZE: usize = 24;
    let mut write_value = [0u8; VALUE_SIZE];
    for i in 0..m {
        let mut write_values = Vec::with_capacity(n + 1);
        for j in 0..(n + 1) {
            if j == 0 {
                write_value[0] = 0 as u8;
                write_value[1] = i as u8;
                write_value[2] = i as u8;
            } else {
                write_value[0] = u_mat[i][j - 1] as u8;
                write_value[1] = p_mat[i][j - 1] as u8;
                write_value[2] = q_mat[i][j - 1] as u8;
            }

            if j == n {
                write_value[3] = i as u8;
                write_value[4] = i as u8;
            } else {
                write_value[3] = pr_mat[i][j] as u8;
                write_value[4] = qr_mat[i][j] as u8;
            }

            write_value[5] = if j == 0 { i } else { d_mat[i][j - 1] as usize } as u8;
            write_values.push(write_value);
        }
        write_value[5] = i as u8;
        write_values.push(write_value);
        pbwt_oram.push(ORAM::<_, VALUE_SIZE>::new_init(
            &write_values,
            PathORAMCreator::with_stash_size(13),
            //LeakyORAMCreator,
            //LinearScanningORAMCreator,
        ));
    }

    println!(
        "ORAM conversion in {} ms",
        (Instant::now() - now).as_millis()
    );
    now = Instant::now();

    /* PBWT querying */
    let mut a_target = vec![0; m];
    let mut d_pre_target = vec![0; m];
    let mut d_post_target = vec![0; m];
    let mut neighbors = vec![vec![0; s]; m];
    a_target[0] = if t[0] == 0 { u_mat[0][n - 1] } else { n };
    if t[0] == 0 {
        d_pre_target[0] = if u_mat[0][n - 1] > 0 { 0 } else { 1 };
        d_post_target[0] = 1;
    } else {
        d_pre_target[0] = if (u_mat[0][n - 1] as usize) < n { 0 } else { 1 };
        d_post_target[0] = 1;
    }

    for i in 1..m {
        let prev_a = a_target[i - 1];
        //assert!((prev_a as usize) < n + 1);

        // Oblivious
        let read_value = pbwt_oram[i].read(prev_a as usize);
        let u_lookup = read_value[0] as usize;
        let p_lookup = read_value[1] as usize;
        let q_lookup = read_value[2] as usize;
        let pr_lookup = read_value[3] as usize;
        let qr_lookup = read_value[4] as usize;

        a_target[i] = if t[i] == 0 {
            u_lookup
        } else {
            u_mat[i][n - 1] + (prev_a - u_lookup)
        };
        //println!("{} {} {} -> {}", u_mat[i][n-1], prev_a, u_lookup, a_target[i]);

        d_pre_target[i] = if t[i] == 0 {
            p_lookup.max(d_pre_target[i - 1])
        } else {
            q_lookup.max(d_pre_target[i - 1])
        };

        d_post_target[i] = if t[i] == 0 {
            pr_lookup.max(d_post_target[i - 1])
        } else {
            qr_lookup.max(d_post_target[i - 1])
        };

        // neighbor finding
        let mut pre_ind = (a_target[i] as i32) - 1;
        let mut pre_div = d_pre_target[i];
        let mut post_ind = a_target[i] as i32;
        let mut post_div = d_post_target[i];
        for l in 0..s {
            let chosen = (pre_ind < 0) || (post_ind < (n as i32) && post_div < pre_div);
            let mut ind = if chosen { post_ind } else { pre_ind };
            neighbors[i][l] = ind as u32;

            if chosen {
                ind += 1;
            } else {
                ind -= 1;
            }

            // Oblivious
            let div;
            if chosen {
                // Oblivious
                let read_value = pbwt_oram[i].read((ind + 1) as usize);
                div = read_value[5] as usize
            } else {
                let read_value = pbwt_oram[i].read((ind + 2) as usize);
                div = read_value[5] as usize
            }

            if chosen {
                post_ind = ind;
                post_div = post_div.max(div);
            } else {
                pre_ind = ind;
                pre_div = pre_div.max(div);
            }
        }
    }

    println!(
        "Neighbor finding in {} ms",
        (Instant::now() - now).as_millis()
    );

    /* Output */
    for j in 0..n {
        if (j + 1 + 5 < a_target[m - 1] as usize) || (j + 1 > a_target[m - 1] as usize + 5) {
            continue;
        }

        print_marker(d_mat[m - 1][j]);

        for i in 0..m {
            print!("{}", x[i][a_mat[m - 1][j] as usize]);
        }
        let mut flag = false;
        for l in 0..s {
            if neighbors[m - 1][l] as usize == j {
                flag = true;
            }
        }
        if flag {
            print!("\tneighbor");
        }
        print!("\n");

        if j + 1 == a_target[m - 1] as usize {
            println!("------------");
            print_marker(d_pre_target[m - 1]);
            for i in 0..m {
                print!("{}", t[i]);
            }
            print!("\ttarget");
            print!("\n");
            print_marker(d_post_target[m - 1]);
            println!("------------");
        }
    }

    for i in 1..m {
        print!("Neighbors at pos {}: ", i);
        for l in 0..s {
            print!("\t{}", neighbors[i][l]);
        }
        print!("\n");
    }
}
