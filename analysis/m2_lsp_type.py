"""
Classify LSP composition near M_2=0 and show which neutralino types populate
the Ωh²≈0 "dip" vs the correct relic density ~0.12 band.

LSP type is inferred from the smallest of |M_1|, |M_2|, |μ| (bino/wino/higgsino),
with "mixed" when the two smallest are within 20%.

Output: /tmp/m2_lsp_type.png
"""
import numpy as np
import uproot
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_GLOB = '/ptmp/jwuerzin/data/18387358/*.root'
N_FILES = 80
OUT = '/tmp/m2_lsp_type.png'
M2_WINDOW = 300
MIX_TOL = 0.8  # smallest/second-smallest ratio above this → "mixed"


def load():
    files = sorted(glob.glob(DATA_GLOB))[:N_FILES]
    M1s, M2s, MUs, Ys, chi = [], [], [], [], []
    for f in files:
        t = uproot.open(f)['susy']
        y = t['MO_Omega'].array(library='np')
        sp = t['SP_m_h'].array(library='np')
        m1 = t['IN_M_1'].array(library='np')
        m2 = t['IN_M_2'].array(library='np')
        mu = t['IN_mu'].array(library='np')
        mc = t['SP_m_chi_10'].array(library='np')
        mask = (y > 0) & (y < 1.0) & (sp != -1)
        M1s.append(m1[mask]); M2s.append(m2[mask]); MUs.append(mu[mask])
        Ys.append(y[mask]); chi.append(mc[mask])
    return tuple(np.concatenate(a) for a in (M1s, M2s, MUs, Ys, chi))


def classify_lsp(M1, M2, MU):
    stack = np.stack([np.abs(M1), np.abs(M2), np.abs(MU)], axis=1)
    winner = stack.argmin(axis=1)  # 0=bino, 1=wino, 2=higgsino
    sorted_idx = np.argsort(stack, axis=1)
    smallest = stack[np.arange(len(stack)), sorted_idx[:, 0]]
    second = stack[np.arange(len(stack)), sorted_idx[:, 1]]
    mixed = (smallest / (second + 1e-6)) > MIX_TOL
    return np.where(mixed, 3, winner)


def main():
    M1, M2, MU, Y, CHI = load()
    label = classify_lsp(M1, M2, MU)
    names = {0: 'bino', 1: 'wino', 2: 'higgsino', 3: 'mixed'}
    near = np.abs(M2) < M2_WINDOW

    print(f'--- near M_2=0 (|M_2|<{M2_WINDOW}) ---')
    for k, v in names.items():
        m = near & (label == k)
        if m.sum() == 0:
            continue
        n_correct = ((Y[m] > 0.08) & (Y[m] < 0.16)).sum()
        print(f'  {v:9s} n={m.sum():6d}  Ωh² median={np.median(Y[m]):.4f}  '
              f'near-correct (0.08<Y<0.16): {n_correct}')

    colors = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red'}
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for k, v in names.items():
        m = near & (label == k)
        ax[0].scatter(M2[m], Y[m], s=3, alpha=0.3, color=colors[k],
                      label=f'{v} (n={m.sum()})')
    ax[0].axhline(0.12, color='k', ls='--', lw=1, label='Ωh²=0.12')
    ax[0].axhspan(0.08, 0.16, color='k', alpha=0.08)
    ax[0].set_xlabel('M_2'); ax[0].set_ylabel('Ωh²')
    ax[0].set_title(f'Ωh² vs M_2 by LSP type (|M_2|<{M2_WINDOW})')
    ax[0].legend(); ax[0].set_xlim(-M2_WINDOW, M2_WINDOW)

    Yt = np.log(Y / 0.12)
    for k, v in names.items():
        m = near & (label == k)
        ax[1].scatter(M2[m], Yt[m], s=3, alpha=0.3, color=colors[k], label=v)
    ax[1].axhline(0, color='k', ls='--', lw=1)
    ax[1].axhspan(np.log(0.08 / 0.12), np.log(0.16 / 0.12), color='k', alpha=0.08)
    ax[1].set_xlabel('M_2'); ax[1].set_ylabel('log(Ωh²/0.12)')
    ax[1].set_title('Log-space: same, by LSP type')
    ax[1].legend(); ax[1].set_xlim(-M2_WINDOW, M2_WINDOW)

    plt.tight_layout(); plt.savefig(OUT, dpi=110)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
