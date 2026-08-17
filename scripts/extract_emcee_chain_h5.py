"""Interleaved (M_1, M_2, mu, Omega) from the RAW emcee chains, burn-in included.

The support figures' emcee reference line must be a genuine partial run: every
proposal charged, repeats (rejections) kept, and the walkers starting where they
actually started rather than pre-converged. The ntuples cannot supply that -- they
hold only post-burn-in rows -- so the line comes from the HDF5 backends while the
band and the support cells continue to come from the ntuples.

Two things here are load-bearing and were verified rather than assumed:

  * `mcmc/chain` carries no column names, and its order is NOT the repo's free
    parameter order. Matching values against the ntuple's first post-burn-in step
    gives dim 0=At, 1=Ab, 2=Atau, 3=M_1, 4=M_2, 5=meL, 6=meR, 7=mu, 8=tanb. The
    naive guess (our free order) would have binned Atau, M_1, M_2 as M_1, M_2, mu.
  * Rows are step-major over the walkers, and a rejected proposal leaves the
    previous position in place, so repeats are present by construction. The
    repeat fraction (0.965) matches 1 - acceptance (0.0386), and rows per ensemble
    minus ntuple rows gives the burn-in exactly, summing to 13,043,712 of
    48,305,152 proposals.

Ensembles ran in parallel, so rows are interleaved round-robin rather than
concatenated: concatenation would describe running one ensemble to completion
before starting the next.

Needs h5py, which lives in the Run3ModelGen environment, not the torch one:
    Run3ModelGen/.pixi/envs/default/bin/python scripts/extract_emcee_chain_h5.py
"""
import glob
import sys

import h5py
import numpy as np

AX = (3, 4, 7)                       # M_1, M_2, mu -- verified, see docstring
CHUNK = 4000                         # steps per read; 4000*256*9*8 = 74 MB
SRC = "/viper/ptmp1/jwuerzin/emcee/neutralino_v4/scan.260731163622.*/chain.h5"
NTUPLE_ROWS = {0: 8740352, 1: 8844032, 2: 8734976, 3: 8942080}
OUT = "/ptmp/jwuerzin/analysis/all_runs/emcee_chain_ordered.npz"

xs, ys, keys, burn_total, raw_total = [], [], [], 0, 0
files = sorted(glob.glob(SRC))
n_ens = len(files)
for e, path in enumerate(files):
    with h5py.File(path, "r") as f:
        g = f["mcmc"]
        it, nw = int(g.attrs["iteration"]), int(g.attrs["nwalkers"])
        Xe = np.empty((it * nw, len(AX)), dtype=np.float32)
        Ye = np.empty(it * nw, dtype=np.float32)
        for s0 in range(0, it, CHUNK):
            s1 = min(s0 + CHUNK, it)
            blk = g["chain"][s0:s1, :, :]
            Xe[s0 * nw:s1 * nw] = blk[:, :, AX].reshape(-1, len(AX))
            Ye[s0 * nw:s1 * nw] = g["blobs"][s0:s1, :]["Omega"].reshape(-1)
    raw = it * nw
    burn = raw - NTUPLE_ROWS[e]
    raw_total += raw
    burn_total += burn
    xs.append(Xe)
    ys.append(Ye)
    keys.append(np.arange(raw, dtype=np.int64) * n_ens + e)
    print(f"  ens{e}: {raw:,} proposals, burn-in {burn:,} "
          f"({burn / raw:.1%})", flush=True)

order = np.argsort(np.concatenate(keys), kind="stable")
X = np.concatenate(xs)[order]
Y = np.concatenate(ys)[order]
np.savez(OUT, X=X, Y=Y, n_rows=np.int64(raw_total),
         burn_rows=np.int64(burn_total))
print(f"wrote {OUT}: {len(Y):,} rows, burn-in {burn_total:,} "
      f"({burn_total / raw_total:.1%}), in-band "
      f"{float((np.abs(Y - 0.12) / 0.12 < 0.1).mean()):.4f}")
