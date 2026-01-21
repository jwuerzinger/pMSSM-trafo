import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

def running_in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False

if running_in_notebook():
    print("Running in Jupyter")
else:
    print("Running as a script")

# Dummy dataset creator:
class Dummy_PMSSMDataset(Dataset):
    def __init__(self, n_samples=10_000):
        super().__init__()
        self.x = torch.randn(n_samples, 19)

        # Dummy "physics-inspired" target
        # Nonlinear function + noise
        self.y = (
            torch.sum(self.x[:, :5] ** 2, dim=1)
            + torch.sin(self.x[:, 5:10]).sum(dim=1)
            + 0.1 * torch.randn(n_samples)
        )
        self.y = self.y.unsqueeze(1)  # (N, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_pmssm_data(n_datasets=-1):
    import uproot, glob, numpy as np, torch

    # Collect all ROOT files in the directory
    files = glob.glob("data/18387358/*.root")
    print(f"Found {len(files)} ROOT files")

    # Open all files
    if n_datasets != -1: 
        print(f"Only using {n_datasets} out of the {len(files)} datasets!!")
        files = files[:n_datasets] # For testing only!
    trees = [uproot.open(f)["susy"] for f in files]


    branches = [
        "IN_meL", "IN_meR", "IN_mtauL", "IN_mtauR",
        "IN_mqL1", "IN_muR", "IN_mdR", "IN_mqL3",
        "IN_mtR", "IN_mbR", "IN_M_1", "IN_M_2",
        "IN_mu", "IN_M_3", "IN_At", "IN_Ab",
        "IN_Atau", "IN_mA", "IN_tanb"
    ]

    X = np.column_stack([
        np.concatenate([t[b].array(library="np") for t in trees])
        for b in branches
    ])

    Y = np.concatenate([t["MO_Omega"].array(library="np") for t in trees])
    plt.hist(Y, bins=20, range=[0.0, 1.0])
    if not running_in_notebook(): plt.savefig('plots/hist_dataset.png')
    else: plt.show()

    # pruning
    mask = (Y != -1.0) & (Y < 1.0)
    X = torch.from_numpy(X[mask]).float()
    Y = torch.from_numpy(Y[mask]).float().unsqueeze(1)

    return X, Y

def make_split(X, train_split=0.9, seed=42):
    N = len(X)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)

    n_train = int(train_split * N)
    idx_train = perm[:n_train]
    idx_val   = perm[n_train:]

    print(f"Have n_train={len(idx_train)}, n_val={len(idx_val)}")
    return idx_train, idx_val

def compute_stats(X, Y, idx_train):
    mean_X = X[idx_train].mean(dim=0)
    std_X  = X[idx_train].std(dim=0) + 1e-8
    mean_Y = Y[idx_train].mean(dim=0)
    std_Y  = Y[idx_train].std(dim=0) + 1e-8

    return mean_X, std_X, mean_Y, std_Y

class PMSSMDataset(Dataset):
    def __init__(self, X, Y, indices, stats, n_samples=None):
        super().__init__()

        mean_X, std_X, mean_Y, std_Y = stats

        X = (X[indices] - mean_X) / std_X
        Y = (Y[indices] - mean_Y) / std_Y

        if n_samples is not None:
            X = X[:n_samples]
            Y = Y[:n_samples]

        self.x = X
        self.y = Y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# class PMSSMTransformer(nn.Module):
#     def __init__(
#         self,
#         n_params=19,
#         d_model=64,
#         nhead=8,
#         num_layers=4,
#         dim_feedforward=256,
#         # dropout=0.1,
#         dropout = 0.0,
#     ):
#         super().__init__()

#         # Embed each scalar parameter
#         self.input_embed = nn.Linear(1, d_model)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.encoder = nn.TransformerEncoder(
#             encoder_layer, num_layers=num_layers
#         )

#         self.regressor = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 1),
#         )

#     def forward(self, x):
#         # x: (batch, 19)
#         x = x.unsqueeze(-1)               # (batch, 19, 1)
#         x = self.input_embed(x)           # (batch, 19, d_model)

#         x = self.encoder(x)               # (batch, 19, d_model)

#         x = x.mean(dim=1)                 # pool over parameters
#         y = self.regressor(x)             # (batch, 1)

#         return y

class PMSSMTransformer(nn.Module):
    """
    Improved transformer with positional encoding to preserve feature order.

    Pros: Maintains feature identity, learns interactions
    Cons: Still might be overkill for small feature set
    """
    def __init__(
        self,
        n_params=19,
        d_model=64,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # Embed each scalar parameter
        self.input_embed = nn.Linear(1, d_model)

        # Learnable positional encoding for each feature
        self.pos_encoding = nn.Parameter(torch.randn(1, n_params, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Use CLS token instead of mean pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (batch, 19)
        batch_size = x.shape[0]

        x = x.unsqueeze(-1)               # (batch, 19, 1)
        x = self.input_embed(x)           # (batch, 19, d_model)

        # Add positional encoding to distinguish features
        x = x + self.pos_encoding        # (batch, 19, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 20, d_model)

        x = self.encoder(x)               # (batch, 20, d_model)

        # Use CLS token for prediction
        x = x[:, 0]                       # (batch, d_model)
        y = self.regressor(x)             # (batch, 1)
        return y

class PMSSMFeedForward(nn.Module):
    def __init__(
        self,
        n_params=19,
        d_model=64,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # Embed each scalar parameter
        self.input_embed = nn.Linear(1, d_model)

        # Build a stack of fully connected layers
        layers = []
        in_features = n_params * d_model
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, dim_feedforward))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout))
            in_features = dim_feedforward
        self.fc_layers = nn.Sequential(*layers)

        # Output layer
        self.regressor = nn.Linear(dim_feedforward, 1)

    def forward(self, x):
        # x: (batch, n_params)
        x = x.unsqueeze(-1)              # (batch, n_params, 1)
        x = self.input_embed(x)          # (batch, n_params, d_model)
        x = x.flatten(start_dim=1)       # (batch, n_params * d_model)
        x = self.fc_layers(x)            # (batch, dim_feedforward)
        y = self.regressor(x)            # (batch, 1)
        return y

def train_with_validation(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device="cpu",
    epochs=30,
    early_stopping=True,
    patience=500,        # early stopping patience
):
    model.to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    counter = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE = {train_loss:.6f} | "
            f"Val MSE = {val_loss:.6f}"
        )

        if early_stopping:
            # ---- Early Stopping Check ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()  # save best weights
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

    # ---- Load best weights before returning ----
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses

def compare_random_predictions(model, stats, subset, mode='validation', device="cpu", n_points=3):
    model.eval()
    model.to(device)

    if mode == 'validation':
        print("\nComparison on random validation points:")
    elif mode == 'train':
        print("\nComparison on random training points:")
    else: raise ValueError("Unsupported mode! Should be validation, train.")
        
    # print(len(dataset), n_points)
    indices = random.sample(range(len(subset)), n_points)

    print("-" * 90)
    print(f"{'Index':>6} | {'True Ωh² (norm.)':>20} | {'Predicted Ωh² (norm.)':>23} | {'True Ωh²':>12} | {'Predicted Ωh²':>15}")
    print("-" * 90)

    with torch.no_grad():
        for idx in indices:
            x, y_true = subset[idx]
            x = x.unsqueeze(0).to(device)      # (1, 19)
            y_pred = model(x).cpu().item()

            # revert normalisation:
            # Y_norm = (Y - self.mean_Y) / self.std_Y
            mean_X, std_X, mean_Y, std_Y = stats
            y_true_nonorm = y_true * std_Y + mean_Y
            y_pred_nonorm = y_pred * std_Y + mean_Y

            print(
                f"{idx:6d} | "
                f"{y_true.item():20.6f} | "
                f"{y_pred:23.6f} | "
                f"{y_true_nonorm.item():12.6f} | "
                f"{y_pred_nonorm.item():15.6f}"
            )

    print("-" * 90)

def is_transformer(model: nn.Module) -> bool:
    return any(
        isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoderLayer))
        for m in model.modules()
    )

def scatter_true_vs_pred(
    model,
    stats,
    subset,
    mode="validation",
    device="cpu",
    denormalize=True,
):
    model.eval()
    model.to(device)

    if mode == "validation":
        title = "Validation set"
    elif mode == "train":
        title = "Training set"
    else:
        raise ValueError("Unsupported mode! Should be validation or train.")

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for x, y_true in subset:
            x = x.unsqueeze(0).to(device)   # (1, n_features)
            y_pred = model(x).cpu().item()

            if denormalize:
                mean_X, std_X, mean_Y, std_Y = stats
                y_true_val = (y_true * std_Y + mean_Y).item()
                y_pred_val = y_pred * std_Y + mean_Y
            else:
                y_true_val = y_true.item()
                y_pred_val = y_pred

            y_true_all.append(y_true_val)
            y_pred_all.append(y_pred_val)

    # --- plot ---
    plt.figure()
    plt.scatter(y_true_all[:10_000], y_pred_all[:10_000], alpha=0.5, color = 'orange' if mode == 'validation' else None)
    plt.plot(
        [min(y_true_all), max(y_true_all)],
        [min(y_true_all), max(y_true_all)],
        linestyle='--',
        color='grey'
    )
    plt.xlabel("True Ωh²")
    plt.ylabel("Predicted Ωh²")
    plt.title(f"True vs Predicted Ωh² ({title})")
    plt.tight_layout()
    # plt.show()
    modelname = "transformer" if is_transformer(model) else "MLP"
    if not running_in_notebook(): plt.savefig(f"plots/{modelname}_true_vs_pred_{mode}.png")
    else: plt.show()

def hist_true_vs_pred(
    model,
    stats,
    subset,
    mode="validation",
    device="cpu",
    denormalize=True,
):
    model.eval()
    model.to(device)

    if mode == "validation":
        title = "Validation set"
    elif mode == "train":
        title = "Training set"
    else:
        raise ValueError("Unsupported mode! Should be validation or train.")

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for x, y_true in subset:
            x = x.unsqueeze(0).to(device)
            y_pred = model(x).cpu().item()

            if denormalize:
                mean_X, std_X, mean_Y, std_Y = stats
                y_true_val = (y_true * std_Y + mean_Y).item()
                y_pred_val = y_pred * std_Y + mean_Y
            else:
                y_true_val = y_true.item()
                y_pred_val = y_pred

            y_true_all.append(y_true_val)
            y_pred_all.append(y_pred_val)

    # --------------------------------------------------
    # 2D histogram
    # --------------------------------------------------

    y_true_arr = np.asarray(y_true_all, dtype=np.float64).reshape(-1)
    y_pred_arr = np.asarray(y_pred_all, dtype=np.float64).reshape(-1)

    plt.figure()
    plt.hist2d(
        y_true_arr,
        y_pred_arr,
        bins=30,
        cmap="inferno",
    )
    plt.colorbar(label="Counts")

    # y = x reference line
    vmin = min(y_true_arr.min(), y_pred_arr.min())
    vmax = max(y_true_arr.max(), y_pred_arr.max())
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="white")

    plt.xlabel("True Ωh²")
    plt.ylabel("Predicted Ωh²")
    modelname = "transformer" if is_transformer(model) else "MLP"
    plt.title(f"True vs Predicted Ωh² ({modelname} {title})")
    plt.tight_layout()

    if not running_in_notebook():
        plt.savefig(f"plots/{modelname}_hist_true_vs_pred_{mode}.png")
    else:
        plt.show()


def plot_losses(train_losses, val_losses, model):
    def rolling_average(x, window=30):
        x = np.asarray(x)
        return np.convolve(x, np.ones(window) / window, mode="valid")
    
    plt.figure()
    plt.plot(rolling_average(train_losses), label="Train MSE")
    plt.plot(rolling_average(val_losses), label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    modelname = "transformer" if is_transformer(model) else "MLP"
    plt.title(f"{modelname} Training for pMSSM Relic Density")
    plt.yscale('log')

    if not running_in_notebook(): plt.savefig(f"plots/losses_{modelname}.png")
    else: plt.show()