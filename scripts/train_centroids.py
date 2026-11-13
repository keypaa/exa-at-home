# scripts/train_centroids.py
"""MiniBatchKMeans on a sample -> centroids.npy. Cloud-only (needs real vecs).

Usage: python scripts/train_centroids.py --vecs V --k K --sample S --out centroids.npy
"""

import argparse


def main(
    vecs_path: str,
    k: int,
    sample: int,
    out: str,
    seed: int = 0,
    verbose: bool = False,
    n_init: int = 3,
    max_iter: int = 100,
):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    vecs = np.load(vecs_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    take = rng.choice(len(vecs), size=min(sample, len(vecs)), replace=False)
    if verbose:
        print(
            f"kmeans: n={len(vecs)} sample={len(take)} k={k} batch=10000 "
            f"n_init={n_init} max_iter={max_iter}",
            flush=True,
        )
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=10_000,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        verbose=1 if verbose else 0,
    )
    km.fit(vecs[take])
    np.save(out, km.cluster_centers_.astype(np.float32))
    if verbose:
        print(
            f"kmeans: wrote {out} shape={km.cluster_centers_.shape} "
            f"inertia={km.inertia_:.3g}",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--sample", type=int, default=500_000)
    ap.add_argument(
        "--n-init",
        type=int,
        default=3,
        help="MiniBatchKMeans restarts; 1 is ~3x faster, measure centroid quality",
    )
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="progress output (sklearn per-init lines + summary)",
    )
    a = ap.parse_args()
    main(
        a.vecs,
        a.k,
        a.sample,
        a.out,
        n_init=a.n_init,
        max_iter=a.max_iter,
        verbose=a.verbose,
    )
