"""Compare two cross_env.py outputs. usage: python -I cross_env_diff.py A.npz B.npz"""
import sys

import numpy as np

a, b = (np.load(p, allow_pickle=True) for p in sys.argv[1:3])
print("A versions:", list(a["versions"]), "| B versions:", list(b["versions"]))
d = np.abs(a["latent"] - b["latent"])
print(f"VAE latent: max abs diff {d.max():.6f}, mean {d.mean():.6f}, latent std {a['latent'].std():.3f}")
for k in ("opt_sq", "opt_wide"):
    x, y = a[k].ravel(), b[k].ravel()
    cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    print(f"optimizer {k}: cosine {cos:.5f}, max abs diff {np.abs(x - y).max():.5f}, perturbation L-inf {np.abs(x).max():.4f} vs {np.abs(y).max():.4f}")
t = np.abs(a["sd_target"].astype(int) - b["sd_target"].astype(int))
print(f"SD target image (same seed): max diff {t.max()}, mean {t.mean():.2f}, identical pixels {(t == 0).mean():.3f}")
