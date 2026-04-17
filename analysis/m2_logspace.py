"""
Compare raw vs log-transformed target Ωh² around M_2=0.

The pipeline trains on Y_t = log(Y / 0.12). The log transform converts the
wino-pole at M_2≈0 (where Ωh² → 0) into a sharp log-space spike with much
larger local variance — which is what MC-Dropout uncertainty surfaces and
the GP's homoscedastic likelihood hides.

Output: /tmp/m2_logspace.png
"""
import numpy as np
import uproot
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_GLOB = '/ptmp/jwuerzin/data/18387358/*.root'
N_FILES = 80
TRUE_VALUE = 0.12
OUT = '/tmp/m2_logspace.png'


def load():
    files = sorted(glob.glob(DATA_GLOB))[:N_FILES]
    M2s, Ys = [], []
    for f in files:
        t = uproot.open(f)['susy']
        y = t['MO_Omega'].array(library='np')
        sp = t['SP_m_h'].array(library='np')
        m2 = t['IN_M_2'].array(library='np')
        mask = (y > 0) & (y < 1.0) & (sp != -1)
        M2s.append(m2[mask]); Ys.append(y[mask])
    return np.concatenate(M2s), np.concatenate(Ys)


def main():
    M2, Y = load()
    Yt = np.log(Y / TRUE_VALUE)
    print(f'N: {len(Y)}')

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].scatter(M2, Y, s=1, alpha=0.12)
    axes[0, 0].axhline(TRUE_VALUE, color='r', ls='--', lw=1)
    axes[0, 0].set_xlabel('M_2'); axes[0, 0].set_ylabel('Ωh² (raw)')
    axes[0, 0].set_title('Raw target Y vs M_2')

    axes[0, 1].scatter(M2, Yt, s=1, alpha=0.12)
    axes[0, 1].axhline(0, color='r', ls='--', lw=1, label='Y=0.12 (obs)')
    axes[0, 1].set_xlabel('M_2'); axes[0, 1].set_ylabel('Y_t = log(Y/0.12)')
    axes[0, 1].set_title('LOG-transformed target Y_t vs M_2 (model-space)')
    axes[0, 1].legend()

    bins = np.linspace(-2000, 2000, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(M2, bins) - 1
    std_yt = np.array([Yt[idx == i].std() if (idx == i).sum() > 10 else np.nan
                       for i in range(len(centers))])
    std_y = np.array([Y[idx == i].std() if (idx == i).sum() > 10 else np.nan
                      for i in range(len(centers))])
    axes[1, 0].plot(centers, std_yt, 'o-', color='C1', label='std(Y_t)')
    axes[1, 0].plot(centers, std_y, 's-', color='C0', label='std(Y raw)')
    axes[1, 0].set_xlabel('M_2 bin center'); axes[1, 0].set_ylabel('within-bin std')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_title('Within-bin target spread')

    near = np.abs(M2) < 200
    far = np.abs(M2) >= 500
    axes[1, 1].hist(Yt[near], bins=60, alpha=0.6, density=True,
                    label=f'|M_2|<200 std={Yt[near].std():.2f}')
    axes[1, 1].hist(Yt[far], bins=60, alpha=0.6, density=True,
                    label=f'|M_2|>=500 std={Yt[far].std():.2f}')
    axes[1, 1].set_xlabel('Y_t'); axes[1, 1].legend()
    axes[1, 1].set_title('Y_t distribution near vs far from M_2=0')

    plt.tight_layout(); plt.savefig(OUT, dpi=110)
    print(f'saved {OUT}')
    print(f'std(Y_t) |M_2|<200  = {Yt[near].std():.3f}')
    print(f'std(Y_t) |M_2|>=500 = {Yt[far].std():.3f}')
    print(f'mean(Y_t) near = {Yt[near].mean():.3f}, far = {Yt[far].mean():.3f}')


if __name__ == '__main__':
    main()
