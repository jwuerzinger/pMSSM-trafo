"""
test_tabpfn.py - Evaluate TabPFN on the pMSSM regression problem.

Phase 1: Synthetic dummy dataset (sanity check)
Phase 2: Real pMSSM data with comparison to RandomForest and Transformer baselines
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor

from pmssm.data import load_pmssm_data, transform_y, inverse_transform_y
from pmssm.models import PMSSMTransformerTabular


# ============================================================
# Transformer helper
# ============================================================
def train_transformer(X_train, y_train, X_val, y_val, device="cuda:0",
                      epochs=2000, patience=200, lr=3e-4):
    """Train a PMSSMTransformerTabular with z-score normalization and early stopping."""
    # Z-score normalization (computed from training set)
    X_tr_t = torch.from_numpy(X_train).float()
    X_val_t = torch.from_numpy(X_val).float()
    y_tr_t = torch.from_numpy(y_train).float().unsqueeze(1)
    y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)

    mean_X = X_tr_t.mean(dim=0)
    std_X = X_tr_t.std(dim=0) + 1e-8
    X_tr_norm = (X_tr_t - mean_X) / std_X
    X_val_norm = (X_val_t - mean_X) / std_X

    train_ds = TensorDataset(X_tr_norm, y_tr_t)
    val_ds = TensorDataset(X_val_norm, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    model = PMSSMTransformerTabular(
        d_model=128, nhead=4, num_layers=3, dim_feedforward=512, dropout=0.1,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_losses.append(criterion(model(X_b), y_b).item() * len(X_b))
        val_loss = sum(val_losses) / len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Restore best model and predict
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(X_val_norm), 1024):
            X_b = X_val_norm[start:start+1024].to(device)
            all_preds.append(model(X_b).cpu())
    y_pred = torch.cat(all_preds, dim=0).squeeze().numpy()

    return y_pred, epoch + 1


# ============================================================
# Phase 1: Dummy Example
# ============================================================
def phase1_dummy():
    print("=" * 60)
    print("Phase 1: Synthetic Dummy Dataset (19 features)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    n_train, n_test = 2000, 500
    n_features = 19

    X_train = rng.randn(n_train, n_features)
    X_test = rng.randn(n_test, n_features)

    # Non-linear target: sum of squares + interactions + noise
    def target_fn(X):
        return (
            np.sum(X[:, :5] ** 2, axis=1)
            + 2.0 * X[:, 0] * X[:, 1]
            - 1.5 * np.sin(X[:, 2] * 3)
            + 0.5 * np.sum(X[:, 10:15], axis=1)
        )

    y_train = target_fn(X_train) + 0.1 * rng.randn(n_train)
    y_test = target_fn(X_test) + 0.1 * rng.randn(n_test)

    # TabPFN
    print("\nFitting TabPFN...")
    t0 = time.time()
    model = TabPFNRegressor(device="cuda:0")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    dt = time.time() - t0
    r2 = r2_score(y_test, y_pred)
    print(f"  TabPFN  R² = {r2:.4f}  ({dt:.1f}s)")

    # RandomForest baseline
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    dt = time.time() - t0
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f"  RF      R² = {r2_rf:.4f}  ({dt:.1f}s)")

    # Transformer
    print("  Fitting Transformer...")
    t0 = time.time()
    y_pred_tr, n_epochs = train_transformer(X_train, y_train, X_test, y_test)
    dt = time.time() - t0
    r2_tr = r2_score(y_test, y_pred_tr)
    print(f"  Trafo   R² = {r2_tr:.4f}  ({dt:.1f}s, {n_epochs} epochs)")

    print()


# ============================================================
# Phase 2: Real pMSSM Data
# ============================================================
def phase2_pmssm():
    print("=" * 60)
    print("Phase 2: Real pMSSM Data")
    print("=" * 60)

    # Load data
    print("\nLoading pMSSM data...")
    X, Y = load_pmssm_data(n_datasets=-1, plot_dir="plots", target="DMRD")
    Y_t = transform_y(Y, target="DMRD").squeeze()  # log(Omega/0.12)
    X_np = X.numpy()
    Y_np = Y_t.numpy()

    print(f"  Total samples: {len(X_np)}")
    print(f"  Features: {X_np.shape[1]}")
    print(f"  Target (transformed) range: [{Y_np.min():.2f}, {Y_np.max():.2f}]")

    # Train/val split (80/20, reproducible)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X_np))
    n_train = int(0.8 * len(X_np))
    idx_train, idx_val = perm[:n_train], perm[n_train:]

    X_train, X_val = X_np[idx_train], X_np[idx_val]
    y_train, y_val = Y_np[idx_train], Y_np[idx_val]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # TabPFN context window limits: 10k training + 267k val OOMs on a single GPU.
    # Keep training sizes where TabPFN can realistically succeed.
    train_sizes = [1_000, 2_000, 5_000]
    train_sizes = [s for s in train_sizes if s <= len(X_train)]

    results = {}

    # Pre-compute physical-space validation targets (needed by all models)
    y_val_phys = inverse_transform_y(
        torch.from_numpy(y_val).float(), target="DMRD"
    ).numpy()

    for n in train_sizes:
        print(f"\n--- Training set size: {n} ---")
        X_tr = X_train[:n]
        y_tr = y_train[:n]

        # TabPFN
        print("  Fitting TabPFN...")
        t0 = time.time()
        try:
            model = TabPFNRegressor(device="cuda:0")
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            dt = time.time() - t0

            r2_t = r2_score(y_val, y_pred)
            y_pred_phys = inverse_transform_y(
                torch.from_numpy(y_pred).float(), target="DMRD"
            ).numpy()
            r2_phys = r2_score(y_val_phys, y_pred_phys)

            print(f"  TabPFN  R²(transformed) = {r2_t:.4f}, "
                  f"R²(physical) = {r2_phys:.4f}  ({dt:.1f}s)")
            results[f"tabpfn_{n}"] = {
                "y_pred_phys": y_pred_phys, "r2_t": r2_t, "r2_phys": r2_phys,
            }
        except Exception as e:
            print(f"  TabPFN failed with n={n}: {e}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                print("  CUDA state is corrupted — skipping TabPFN for remaining sizes.")

        # RandomForest
        print("  Fitting RandomForest...")
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred_rf = rf.predict(X_val)
        dt = time.time() - t0

        r2_rf_t = r2_score(y_val, y_pred_rf)
        y_pred_rf_phys = inverse_transform_y(
            torch.from_numpy(y_pred_rf).float(), target="DMRD"
        ).numpy()
        r2_rf_phys = r2_score(y_val_phys, y_pred_rf_phys)
        print(f"  RF      R²(transformed) = {r2_rf_t:.4f}, "
              f"R²(physical) = {r2_rf_phys:.4f}  ({dt:.1f}s)")
        results[f"rf_{n}"] = {
            "y_pred_phys": y_pred_rf_phys, "r2_t": r2_rf_t, "r2_phys": r2_rf_phys,
        }

        # Transformer
        print("  Fitting Transformer...")
        t0 = time.time()
        try:
            y_pred_tr, n_epochs = train_transformer(
                X_tr, y_tr, X_val, y_val, device="cuda:0",
            )
            dt = time.time() - t0

            r2_tr_t = r2_score(y_val, y_pred_tr)
            y_pred_tr_phys = inverse_transform_y(
                torch.from_numpy(y_pred_tr).float(), target="DMRD"
            ).numpy()
            r2_tr_phys = r2_score(y_val_phys, y_pred_tr_phys)

            print(f"  Trafo   R²(transformed) = {r2_tr_t:.4f}, "
                  f"R²(physical) = {r2_tr_phys:.4f}  ({dt:.1f}s, {n_epochs} epochs)")
            results[f"trafo_{n}"] = {
                "y_pred_phys": y_pred_tr_phys, "r2_t": r2_tr_t, "r2_phys": r2_tr_phys,
            }
        except Exception as e:
            print(f"  Transformer failed with n={n}: {e}")

    # Scatter plots: one row per training size, TabPFN | RF | Transformer columns
    plotted_sizes = [n for n in train_sizes if f"rf_{n}" in results]
    if not plotted_sizes:
        print("\nNo results to plot.")
        return

    n_rows = len(plotted_sizes)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    model_keys = [
        ("tabpfn", "TabPFN"),
        ("rf", "RandomForest"),
        ("trafo", "Transformer"),
    ]

    for row, n in enumerate(plotted_sizes):
        for col, (key, label) in enumerate(model_keys):
            ax = axes[row, col]
            result_key = f"{key}_{n}"

            if result_key in results:
                yp = results[result_key]["y_pred_phys"]
                ax.scatter(y_val_phys, yp, alpha=0.1, s=2)
                lo = min(y_val_phys.min(), yp.min())
                hi = max(y_val_phys.max(), yp.max())
                ax.plot([lo, hi], [lo, hi], "r--", lw=1)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal")
                r2 = r2_score(y_val_phys, yp)
                ax.set_title(f"{label} (n={n})\nR² = {r2:.4f}")
            else:
                ax.text(0.5, 0.5, f"{label} failed\n(n={n})",
                        ha="center", va="center", fontsize=14, color="gray",
                        transform=ax.transAxes)
                ax.set_title(f"{label} (n={n})")

            ax.set_xlabel("True Ωh²")
            ax.set_ylabel("Predicted Ωh²")

    plt.tight_layout()
    out_path = "plots/tabpfn_vs_rf_vs_trafo_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nScatter plot saved to {out_path}")


if __name__ == "__main__":
    import subprocess, sys

    # Each phase runs in its own subprocess to isolate CUDA state.
    # A CUDA illegal-memory-access error poisons the entire process,
    # so we cannot run both phases sequentially in the same process.
    if len(sys.argv) > 1:
        if sys.argv[1] == "phase1":
            phase1_dummy()
        elif sys.argv[1] == "phase2":
            phase2_pmssm()
    else:
        for phase in ["phase1", "phase2"]:
            print(f"\n{'='*60}")
            print(f"Launching {phase} in subprocess...")
            print(f"{'='*60}\n")
            ret = subprocess.run(
                [sys.executable, __file__, phase],
                env={**__import__("os").environ},
            )
            if ret.returncode != 0:
                print(f"\n{phase} exited with code {ret.returncode}")
