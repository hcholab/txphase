use oram_sgx::*;
use rand::RngCore;
use std::borrow::Cow;
use std::time::Instant;

fn main() {
    const VALUE_SIZE: usize = 8;
    let n_values = 1 << 18;
    let n_reads = 1000;

    let mut values = vec![[0u8; VALUE_SIZE]; n_values];
    let mut counter = 0u8;

    for value in values.iter_mut() {
        for byte in value.iter_mut() {
            *byte = counter;
            counter = counter.wrapping_add(1);
        }
    }
    println!("=== Leaky ===");
    {
        let start = Instant::now();
        let mut oram = ORAM::<_, VALUE_SIZE>::new_init(
            values.iter().map(|v| Cow::Borrowed(v)),
            LeakyORAMCreator,
        );
        let init_time = (Instant::now() - start).as_millis();
        println!("Initialization time: {} ms", init_time);

        let start = Instant::now();
        for _ in 0..n_reads {
            let index = rand::thread_rng().next_u64() as usize % n_values;
            let result = oram.read(index);
            assert_eq!(result, values[index]);
        }
        let avg_read = (Instant::now() - start).as_nanos() as f64 / n_reads as f64 / 1000000.;
        println!("Avg. read time: {:.5} ms", avg_read);
    }

    println!("=== Linear Scan ===");
    {
        let start = Instant::now();
        let mut oram = ORAM::<_, VALUE_SIZE>::new_init(
            values.iter().map(|v| Cow::Borrowed(v)),
            LinearScanningORAMCreator,
        );
        let init_time = (Instant::now() - start).as_millis();
        println!("Initialization time: {} ms", init_time);

        let start = Instant::now();
        for _ in 0..n_reads {
            let index = rand::thread_rng().next_u64() as usize % n_values;
            let result = oram.read(index);
            assert_eq!(result, values[index]);
        }
        let avg_read = (Instant::now() - start).as_nanos() as f64 / n_reads as f64 / 1000000.;
        println!("Avg. read time: {:.5} ms", avg_read);
    }

    println!("=== Path ORAM ===");
    {
        let start = Instant::now();
        let mut oram = ORAM::<_, VALUE_SIZE>::new_init(
            values.iter().map(|v| Cow::Borrowed(v)),
            PathORAMCreator::with_stash_size(13),
        );
        let init_time = (Instant::now() - start).as_millis();
        println!("Initialization time: {} ms", init_time);

        let start = Instant::now();
        for _ in 0..n_reads {
            let index = rand::thread_rng().next_u64() as usize % n_values;
            let result = oram.read(index);
            assert_eq!(result, values[index]);
        }
        let avg_read = (Instant::now() - start).as_nanos() as f64 / n_reads as f64 / 1000000.;
        println!("Avg. read time: {:.5} ms", avg_read);
    }
}
