// crates/ann-core/src/ivf.rs
// M3: centroid routing + inverted-list scan. Bit order is the Task 5
// contract (byte i/8, bit i%8, LSB-first) via quant::binary_dot_packed.

use crate::{lut, quant};

/// Route a query to the `nprobe` nearest centroids (exact float dot over K
/// centroids, partial select). `centroids` is K×`dim` row-major. Returns
/// cluster ids sorted best-first. `nprobe` is clamped to K.
pub fn route(centroids: &[f32], k: usize, dim: usize, q: &[f32], nprobe: usize) -> Vec<u32> {
    assert_eq!(q.len(), dim);
    let mut scored: Vec<(u32, f32)> = (0..k as u32)
        .map(|c| {
            let row = &centroids[c as usize * dim..(c as usize + 1) * dim];
            (c, quant::dot(row, q))
        })
        .collect();
    let nprobe = nprobe.min(k);
    if nprobe < scored.len() {
        scored.select_nth_unstable_by(nprobe, |a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(nprobe);
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().map(|(c, _)| c).collect()
}

/// Linear scan of the selected posting lists, M4 LUT-scored.
/// `codes` is n×32 row-major packed rows; `lists` holds per-cluster row
/// ids. `allow`, when present, must be sorted ascending and is enforced by
/// binary search (the binding sorts once per query). Returns (row id,
/// score) pairs sorted best-first, truncated to `top_k`.
///
/// M4: one 64×16 LUT is built per query (`lut::build`) and each doc is
/// scored with 64 nibble-gather lookups (`lut::score`) instead of the naive
/// 256-iteration `quant::binary_dot_packed` loop. Scores agree within fp
/// reassociation (see `lut::tests`). Non-256-dim queries (unit-test scale
/// only — production dim is always 256) fall back to the naive loop so this
/// fn stays total over any `q.len()`; both scoring fns are kept so the
/// criterion bench can compare M2 vs M4.
pub fn search_lists(
    lists: &[Vec<u32>],
    codes: &[[u8; 32]],
    cluster_ids: &[u32],
    q: &[f32],
    top_k: usize,
    allow: Option<&[u32]>,
) -> Vec<(u32, f32)> {
    let table: Option<[[f32; 16]; 64]> = match q.try_into() {
        Ok(qa) => Some(lut::build(qa)),
        Err(_) => None,
    };
    let mut scored: Vec<(u32, f32)> = Vec::new();
    for &c in cluster_ids {
        for &row in &lists[c as usize] {
            if let Some(a) = allow {
                if a.binary_search(&row).is_err() {
                    continue;
                }
            }
            let s = match &table {
                Some(t) => lut::score(&codes[row as usize], t),
                None => quant::binary_dot_packed(&codes[row as usize], q),
            };
            scored.push((row, s));
        }
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(top_k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(rows: &[[f32; 4]]) -> Vec<[u8; 32]> {
        // Tiny-dim stand-in: only the first 4 bits vary; the rest are 0.
        rows.iter()
            .map(|r| {
                let mut c = [0u8; 32];
                for (i, x) in r.iter().enumerate() {
                    if *x > 0.0 {
                        c[i / 8] |= 1 << (i % 8);
                    }
                }
                c
            })
            .collect()
    }

    #[test]
    fn route_picks_nearest_centroids_best_first() {
        // 2-D centroids; query (1, 0): dot order is c1 > c0 > c2.
        let centroids = vec![0.0, 1.0, 1.0, 0.0, -1.0, 0.0];
        assert_eq!(route(&centroids, 3, 2, &[1.0, 0.0], 2), vec![1, 0]);
        // nprobe > K clamps instead of panicking.
        assert_eq!(route(&centroids, 3, 2, &[1.0, 0.0], 99), vec![1, 0, 2]);
    }

    #[test]
    fn search_lists_ranks_by_binary_dot_and_honors_allow() {
        let codes = packed(&[
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]);
        let lists = vec![vec![0, 2], vec![1]];
        let q = vec![1.0; 256][..4].to_vec();
        let got = search_lists(&lists, &codes, &[0, 1], &q, 10, None);
        assert_eq!(
            got.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        let got = search_lists(&lists, &codes, &[0, 1], &q, 10, Some(&[2]));
        assert_eq!(got.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![2]);
    }
}
