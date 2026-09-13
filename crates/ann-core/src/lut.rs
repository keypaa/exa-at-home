// crates/ann-core/src/lut.rs
// M4: per-query 64×16 LUT dot-product scoring (Exa's length-4 trick).
//
// 256 bits = 64 nibbles. For nibble `g` covering dims `4g..4g+4`, table
// entry `e` (0–15) = Σ bit_j(e)·(+q / −q) over the 4 dims. Scoring a doc is
// then 64 table lookups vs 256 FMAs naive (4x fewer mults,
// register-resident tables).
//
// BIT-ORDER CONTRACT (must match `quant::binarize` — do not change):
// doc bit `i` lives in byte `i/8` at position `i%8`. Nibble `2b` = dims
// `8b..8b+4` = LOW 4 bits of byte `b`; nibble `2b+1` = HIGH 4 bits. In
// `build`, entry bit `j` corresponds to dim `4g+j` sign +1.

/// Per-query tables: 64 nibbles × 16 sign-combos.
pub fn build(q: &[f32; 256]) -> [[f32; 16]; 64] {
    let mut t = [[0.0f32; 16]; 64];
    for g in 0..64 {
        for e in 0..16 {
            let mut s = 0.0;
            for j in 0..4 {
                let v = q[4 * g + j];
                if (e >> j) & 1 == 1 {
                    s += v;
                } else {
                    s -= v;
                }
            }
            t[g][e] = s;
        }
    }
    t
}

/// 2 lookups per byte (low nibble, high nibble), 64 total.
pub fn score(doc: &[u8; 32], t: &[[f32; 16]; 64]) -> f32 {
    let mut s = 0.0;
    for (b, chunk) in doc.iter().enumerate() {
        s += t[2 * b][(chunk & 0x0F) as usize];
        s += t[2 * b + 1][(chunk >> 4) as usize];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_naive() {
        // NOTE vs the plan sketch: `q` is collected into `[f32; 256]` first.
        // The sketch's `&q.try_into()` moves the Vec and then uses `q` again
        // in `binary_dot_packed(&doc, &q)` — that does not compile (use after
        // move). Same construction otherwise: sin sweep, self-query (all
        // positive bits pattern), tolerance 1e-3.
        let q: [f32; 256] = (0..256)
            .map(|i| (i as f32).sin())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let doc = crate::quant::binarize(&q); // self-query: all positive bits pattern
        let t = build(&q);
        let s_lut = score(&doc, &t);
        let s_naive = crate::quant::binary_dot_packed(&doc, &q);
        assert!((s_lut - s_naive).abs() < 1e-3);
    }

    #[test]
    fn lut_matches_naive_random_bits() {
        // Self-query only exercises entries where doc bits agree with the
        // query signs; random doc bytes exercise all 16 entries per nibble
        // (catches nibble-swap / entry-bit-order bugs the self-query misses).
        let mut st: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let q: [f32; 256] =
            std::array::from_fn(|_| (next() as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0);
        let t = build(&q);
        for _ in 0..32 {
            let doc: [u8; 32] = std::array::from_fn(|_| (next() & 0xFF) as u8);
            let s_lut = score(&doc, &t);
            let s_naive = crate::quant::binary_dot_packed(&doc, &q);
            assert!(
                (s_lut - s_naive).abs() < 1e-3,
                "lut {s_lut} vs naive {s_naive}"
            );
        }
    }
}
