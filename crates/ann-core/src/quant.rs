// crates/ann-core/src/quant.rs
/// M1: exact float dot product. Later milestones add binary paths beside this.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 256-dim float row -> 32 bytes, bit i = (x[i] > 0).
///
/// BIT-ORDER CONTRACT (Task 7's LUT depends on this — do not change):
/// bit i lives LSB-first in byte `i / 8`, bit `i % 8`, i.e. element 0 is the
/// LSB of byte 0. Python side this matches
/// `np.packbits((row > 0).astype(np.uint8), bitorder="little")`
/// (NOT packbits' default `bitorder="big"`).
pub fn binarize(row: &[f32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, x) in row.iter().enumerate() {
        if *x > 0.0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Naive packed dot: bit=1 -> +q[i], else -> -q[i]. M4 replaces inner loop with LUT.
pub fn binary_dot_packed(doc: &[u8; 32], q: &[f32]) -> f32 {
    let mut s = 0.0;
    for (i, v) in q.iter().enumerate() {
        if (doc[i / 8] >> (i % 8)) & 1 == 1 {
            s += v;
        } else {
            s -= v;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dot_unit_vectors() {
        assert!((dot(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(dot(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn binarize_sign_bits_lsb_first() {
        // bit i = (x[i] > 0), byte i/8, bit i%8; zero maps to 0 (not > 0).
        let mut row = [0.0f32; 256];
        row[0] = 1.0; // byte 0, bit 0 -> 0x01
        row[7] = 1.0; // byte 0, bit 7 -> 0x80
        row[8] = 1.0; // byte 1, bit 0 -> 0x01
        row[255] = -1.0; // stays 0
        let out = binarize(&row);
        assert_eq!(out[0], 0x81);
        assert_eq!(out[1], 0x01);
        assert_eq!(out[31], 0x00);
        assert!(out[2..31].iter().all(|&b| b == 0));
    }

    #[test]
    fn binary_dot_packed_matches_signed_sum() {
        // doc bits: even i -> 1, odd i -> 0; check against explicit +q[i]/-q[i].
        let mut q = [0.0f32; 256];
        for (i, v) in q.iter_mut().enumerate() {
            *v = i as f32 * 0.01 - 1.0;
        }
        let doc = binarize(
            &(0..256)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect::<Vec<f32>>(),
        );
        let want: f32 = q
            .iter()
            .enumerate()
            .map(|(i, v)| if i % 2 == 0 { *v } else { -*v })
            .sum();
        assert!((binary_dot_packed(&doc, &q) - want).abs() < 1e-3);
    }
}
