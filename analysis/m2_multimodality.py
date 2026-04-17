"""
Investigate how Ωh² depends on M_2 in raw and log-transformed space.

Context: GP candidate uncertainty dips at M_2≈0 while transformer/TabPFN
uncertainty spikes there. This script probes the training-target structure
near M_2=0 to explain the asymmetry.

Output: /tmp/m2_multimodality.png  (raw-Y analysis)
"""
import numpy as np
import uproot
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_GLOB = '/ptmp/jwuerzin/data/18387358/*.root'
N_FILES = 50
OUT = '/tmp/m2_multimodality.png'


def load():
    files = sorted(glob.glob(DATA_GLOB))[:N_FILES]
    M2s, Ys, MUs, M1s = [], [], [], []
    for f in files:
        t = uproot.open(f)['susy']
        y = t['MO_Omega'].array(library='np')
        sp = t['SP_m_h'].array(library='np')
        m2 = t['IN_M_2'].array(library='np')
        mu = t['IN_mu'].array(library='np')
        m1 = t['IN_M_1'].array(library='np')
        mask = (y > 0) & (y < 1.0) & (sp != -1)
        M2s.append(m2[mask]); Ys.append(y[mask])
        MUs.append(mu[mask]); M1s.append(m1[mask])
    return tuple(np.concatenate(a) for a in (M2s, Ys, MUs, M1s))


def main():
    M2, Y, MU, M1 = load()
    print(f'N after filter: {len(Y)}')

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(M2, Y, s=1, alpha=0.15)
    axes[0, 0].set_xlabel('M_2'); axes[0, 0].set_ylabel('Ωh²')
    axes[0, 0].set_title('All data: Ωh² vs M_2')
    axes[0, 0].axhline(0.12, color='r', ls='--', lw=1, label='observed')
    axes[0, 0].legend()

    m = np.abs(M2) < 400
    axes[0, 1].scatter(M2[m], Y[m], s=2, alpha=0.3, c=MU[m], cmap='coolwarm')
    axes[0, 1].set_xlabel('M_2'); axes[0, 1].set_ylabel('Ωh²')
    axes[0, 1].set_title('Zoom |M_2|<400 (color = mu)')
    axes[0, 1].axhline(0.12, color='r', ls='--', lw=1)

    bins = np.linspace(-2000, 2000, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(M2, bins) - 1
    std_y = np.array([Y[idx == i].std() if (idx == i).sum() > 10 else np.nan
                      for i in range(len(centers))])
    med_y = np.array([np.median(Y[idx == i]) if (idx == i).sum() > 10 else np.nan
                      for i in range(len(centers))])
    axes[1, 0].plot(centers, std_y, 'o-', label='std(Ωh²)')
    axes[1, 0].plot(centers, med_y, 's-', label='median(Ωh²)')
    axes[1, 0].set_xlabel('M_2 bin center'); axes[1, 0].set_title('Target spread vs M_2')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    small = np.abs(M2) < 200
    axes[1, 1].hist(Y[small], bins=50, range=(0, 1), alpha=0.6, density=True,
                    label=f'|M_2|<200 (n={small.sum()})')
    axes[1, 1].hist(Y[~small], bins=50, range=(0, 1), alpha=0.6, density=True,
                    label=f'|M_2|>=200 (n={(~small).sum()})')
    axes[1, 1].set_xlabel('Ωh²'); axes[1, 1].set_title('Near vs far from M_2=0')
    axes[1, 1].legend(); axes[1, 1].set_yscale('log')

    plt.tight_layout(); plt.savefig(OUT, dpi=110)
    print(f'saved {OUT}')
    print(f'std(Y) |M_2|<200: {Y[small].std():.4f}, |M_2|>=200: {Y[~small].std():.4f}')
    print(f'mean(Y) |M_2|<200: {Y[small].mean():.4f}, |M_2|>=200: {Y[~small].mean():.4f}')


if __name__ == '__main__':
    main()
