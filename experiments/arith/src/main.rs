use num_format::{Locale, ToFormattedString};
use rand::RngCore;
use std::hint::black_box;
use tp_fixedpoint::TpFixed64;

type Real = TpFixed64<52>;
const N: usize = 100;

fn add(a: &mut [Real], b: &[Real]) {
    a.iter_mut().zip(b.iter()).for_each(|(a, &b)| *a += b);
}

fn sub(a: &mut [Real], b: &[Real]) {
    a.iter_mut().zip(b.iter()).for_each(|(a, &b)| *a -= b);
}

fn mult(a: &mut [Real], b: &[Real]) {
    a.iter_mut().zip(b.iter()).for_each(|(a, &b)| *a *= b);
}

fn div(a: &mut [Real], b: &[Real]) {
    a.iter_mut().zip(b.iter()).for_each(|(a, &b)| *a /= b);
}

fn bench(count: usize, a: &[Real], b: &[Real], f: impl Fn(&mut [Real], &[Real])) {
    let mut a = a.to_owned();
    let b = b.to_owned();
    let t = std::time::Instant::now();
    for _ in 0..count / a.len() {
        f(black_box(&mut a), black_box(&b));
    }
    let ms = t.elapsed().as_millis();
    println!("count: {}", count.to_formatted_string(&Locale::en));
    println!("\ttime: {} ms", ms.to_formatted_string(&Locale::en));
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut a = [Real::ZERO; N];
    let mut b = [Real::ZERO; N];
    a.iter_mut()
        .for_each(|v| *v = Real::protect_i64(rng.next_u64() as i64));
    b.iter_mut()
        .for_each(|v| *v = Real::protect_i64(rng.next_u64() as i64));

    println!("### Addition experiments ###");
    let n_add_d = 19251708878;
    bench(n_add_d, &a, &b, add);
    let n_add_l = 1062560831759;
    bench(n_add_l, &a, &b, add);

    println!("### Subtraction experiments ###");
    let n_sub_d = 3059460;
    bench(n_sub_d, &a, &b, sub);
    let n_sub_l = 115526221459;
    bench(n_sub_l, &a, &b, sub);

    println!("### Multiplication experiments ###");
    let n_mult_d = 3551632629;
    bench(n_mult_d, &a, &b, mult);
    let n_mult_l = 981993185238;
    bench(n_mult_l, &a, &b, mult);

    println!("### Division experiments ###");
    let n_div_d = 670891;
    bench(n_div_d, &a, &b, div);
}
