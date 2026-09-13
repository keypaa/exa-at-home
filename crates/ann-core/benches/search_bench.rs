// crates/ann-core/benches/search_bench.rs
// M4 (Task 7): naive `binary_dot_packed` vs LUT scoring, same docs/query.
// Task 12 extends this file (single-query + concurrency 1/8/64).
//
// 2000 docs × 32B codes, random bits from a tiny in-Rust xorshift (no
// external files, deterministic seed so runs are comparable). The query is
// a fixed pseudo-random [f32; 256]. Each bench iteration re-builds the LUT
// (bench `lut_scan`) or not (bench `naive_scan`), so the comparison
// includes the per-query build cost — the honest end-to-end number.

use ann_core::{lut, quant};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

const N_DOCS: usize = 2000;

struct Xor64(u64);

impl Xor64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn fixture() -> (Vec<[u8; 32]>, [f32; 256]) {
    let mut rng = Xor64(0x9E3779B97F4A7C15);
    let mut docs = Vec::with_capacity(N_DOCS);
    for _ in 0..N_DOCS {
        let mut c = [0u8; 32];
        for b in c.iter_mut() {
            *b = (rng.next() & 0xFF) as u8;
        }
        docs.push(c);
    }
    let q: [f32; 256] =
        std::array::from_fn(|_| (rng.next() as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0);
    (docs, q)
}

fn scan_naive(docs: &[[u8; 32]], q: &[f32; 256]) -> f32 {
    docs.iter().map(|d| quant::binary_dot_packed(d, q)).sum()
}

fn scan_lut(docs: &[[u8; 32]], q: &[f32; 256]) -> f32 {
    let t = lut::build(q);
    docs.iter().map(|d| lut::score(d, &t)).sum()
}

fn bench_search(c: &mut Criterion) {
    let (docs, q) = fixture();
    // Sanity: both paths must agree before we time anything.
    let a = scan_naive(&docs, &q);
    let b = scan_lut(&docs, &q);
    assert!(
        (a - b).abs() / a.abs().max(1.0) < 1e-4,
        "LUT disagrees with naive: {a} vs {b}"
    );

    c.bench_function("binary_dot_packed_scan_2k", |ben| {
        ben.iter(|| black_box(scan_naive(black_box(&docs), black_box(&q))))
    });
    c.bench_function("lut_scan_2k", |ben| {
        ben.iter(|| black_box(scan_lut(black_box(&docs), black_box(&q))))
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
