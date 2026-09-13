# scripts/train_centroids.py
"""MiniBatchKMeans on a sample -> centroids.npy. Cloud-only (needs real vecs).

Usage: python scripts/train_centroids.py --vecs V --k K --sample S --out centroids.npy
"""
import argparse


def main(vecs_path: str, k: int, sample: int, out: str, seed: int = 0):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    vecs = np.load(vecs_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    take = rng.choice(len(vecs), size=min(sample, len(vecs)), replace=False)
    km = MiniBatchKMeans(n_clusters=k, batch_size=10_000, n_init=3, random_state=seed)
    km.fit(vecs[take])
    np.save(out, km.cluster_centers_.astype(np.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True); ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--sample", type=int, default=500_000); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.vecs, a.k, a.sample, a.out)
