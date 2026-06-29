#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Neural Pursuit (DNP) — Single-file PyTorch implementation + main()

Implements a DNP-style greedy feature selection for HDLSS data:
 • Greedy addition by input-layer gradient group-norms (L2) averaged across dropout samples
 • Keeps unselected input columns fixed at zero (W_C = 0) during subnetwork training
 • Adagrad optimizer; Xavier init for newly activated feature columns
 • Optional bootstrap stability (Jaccard)

Inputs (CSV):
  - --data     : shape (n_samples, n_features)
  - --features : shape (n_features,) one name per line
  - --labels   : shape (n_samples,) binary labels {0,1}

Primary outputs:
  - selected_genes_dnp.csv     (ordered feature names)
  - dnp_report.txt             (run summary)
  - stability_report.txt       (if --stability-runs > 0)

Additional outputs:
  - data_reduced.csv           (X restricted to selected columns, same sample order)
  - features_reduced.csv       (selected feature names, ordered)
  - indices_reduced.txt        (0-based column indices, ordered)
"""

import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

#np.random.seed(42)
# Optional: sklearn for AUC
try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# -----------------------------
# Utilities
# -----------------------------
def standardize_np(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def xavier_bound(fan_in, fan_out):
    return math.sqrt(6.0 / (fan_in + fan_out))


# -----------------------------
# MLP backbone (exposes first-layer weights)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=(50, 30, 15), dropout_p=0.5):
        super().__init__()
        self.dropout_p = float(dropout_p)

        self.first = nn.Linear(input_dim, hidden_layers[0], bias=True)
        nn.init.zeros_(self.first.weight)
        nn.init.zeros_(self.first.bias)

        blocks = []
        prev = hidden_layers[0]
        for h in hidden_layers[1:]:
            blocks += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout_p)]
            prev = h
        self.hidden = nn.Sequential(*blocks)

        self.out = nn.Linear(prev, 1)  # single logit
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        z = self.first(x)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout_p, training=self.training)
        if len(self.hidden) > 0:
            z = self.hidden(z)
        logit = self.out(z)
        return logit.squeeze(1)

    @property
    def W1(self):
        # shape: [hidden1, input_dim]
        return self.first.weight


# -----------------------------
# DNP Classifier
# -----------------------------
class DNPClassifier:
    """
    PyTorch DNP:
     - Greedy selection via ||grad(W1[:, j])||_2 (averaged across dropout passes)
     - W_C=0 for unselected columns during training
     - Adagrad optimizer
     - Xavier init for newly activated column
    """

    def __init__(
        self,
        input_dim,
        hidden_layers=(50, 30, 15),
        k_features=25,
        dropout_p=0.5,
        n_dropout_samples=12,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=None,
        max_epochs_per_iter=200,
        early_stopping_patience=20,
        device="auto",
        seed=42,
    ):
        self.input_dim = int(input_dim)
        self.hidden_layers = tuple(hidden_layers)
        self.k_features = int(k_features)
        self.dropout_p = float(dropout_p)
        self.n_dropout_samples = int(n_dropout_samples)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = batch_size
        self.max_epochs_per_iter = int(max_epochs_per_iter)
        self.early_stopping_patience = int(early_stopping_patience)

        # Device resolution
        self.device = device
        if self.device in (None, "auto"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.seed = int(seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.model = MLP(self.input_dim, self.hidden_layers, dropout_p=self.dropout_p).to(self.device)
        self.selected_mask_ = np.zeros(self.input_dim, dtype=bool)  # True if selected
        self.ranking_ = []
        self.scaler_ = None  # (mu, sd)

    # -------- data helpers --------
    def _to_tensor(self, X, y=None):
        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if y is None:
            return Xt, None
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        return Xt, yt

    def _loader(self, X, y):
        if self.batch_size is None or self.batch_size >= len(X):
            ds = torch.utils.data.TensorDataset(X, y)
            return torch.utils.data.DataLoader(ds, batch_size=len(X), shuffle=False)
        ds = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    # -------- W_C = 0 enforcement --------
    def _zero_candidate_cols(self):
        with torch.no_grad():
            W = self.model.W1  # [hidden1, input_dim]
            mask = torch.as_tensor(~self.selected_mask_, device=W.device)  # True = candidate
            W[:, mask] = 0.0

    def _after_step_rezero(self):
        self._zero_candidate_cols()

    # -------- train one subnetwork --------
    def _train_subnetwork(self, Xtr, ytr, Xval, yval):
        self.model.train()
        opt = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = nn.BCEWithLogitsLoss()
        loader = self._loader(Xtr, ytr)

        best_val = float('inf')
        best_state = None
        wait = 0

        for _epoch in range(self.max_epochs_per_iter):
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = bce(logits, yb)
                loss.backward()

                # Keep candidates frozen
                with torch.no_grad():
                    W = self.model.W1
                    mask_cand = torch.as_tensor(~self.selected_mask_, device=W.device)
                    if W.grad is not None:
                        W.grad[:, mask_cand] = 0.0

                opt.step()
                self._after_step_rezero()

            # Early stopping on validation loss
            self.model.eval()
            with torch.no_grad():
                vloss = bce(self.model(Xval), yval).item()
            self.model.train()

            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if wait >= self.early_stopping_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._after_step_rezero()

    # -------- candidate scoring: average dropout gradients --------
    def _score_candidates(self, X, y):
        self.model.train()  # enable dropout
        bce = nn.BCEWithLogitsLoss()
        acc = torch.zeros_like(self.model.W1)  # [hidden1, input_dim]

        # multiple gradient samplings with dropout noise
        for _ in range(self.n_dropout_samples):
            self._zero_candidate_cols()
            self.model.zero_grad(set_to_none=True)
            logits = self.model(X)
            loss = bce(logits, y)
            loss.backward()
            acc += self.model.W1.grad.clone()

        # average outside loop
        acc /= float(self.n_dropout_samples)

        # Column-wise L2 norms (feature scores)
        grad_cols = acc.norm(p=2, dim=0).detach().cpu().numpy()  # [input_dim]

        # Guard against non-finite values
        grad_cols = np.where(np.isfinite(grad_cols), grad_cols, -np.inf)

        # Exclude already selected
        grad_cols[self.selected_mask_] = -np.inf

        # Tie-breaking jitter (seeded, tiny; keeps reproducibility but avoids argmax==0 ties)
        rng = np.random.RandomState(self.seed)
        eps = rng.normal(loc=0.0, scale=1e-12, size=grad_cols.shape)
        tie_safe = grad_cols + eps

        j = int(np.argmax(tie_safe))
        score = float(grad_cols[j])
        return j, score, grad_cols

    # -------- public API --------
    def fit(self, X, y, X_val=None, y_val=None, k=None, standardize=True):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        if standardize:
            X, mu, sd = standardize_np(X)
            self.scaler_ = (mu, sd)

        # Small internal validation split (20%) if none provided
        if X_val is None or y_val is None:
            n = len(X)
            m = max(1, int(0.2 * n))
            idx = np.arange(n)
            rng = np.random.RandomState(self.seed)
            rng.shuffle(idx)
            val_idx, tr_idx = idx[:m], idx[m:]
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xval, yval = X[val_idx], y[val_idx]
        else:
            Xtr, ytr = X, y
            Xval, yval = np.asarray(X_val, dtype=np.float32), np.asarray(y_val, dtype=np.float32)
            if standardize and self.scaler_ is not None:
                mu, sd = self.scaler_
                Xval = (Xval - mu) / sd

        Xtr_t, ytr_t = self._to_tensor(Xtr, ytr)
        Xval_t, yval_t = self._to_tensor(Xval, yval)

        self.ranking_ = []
        self.selected_mask_[:] = False
        target_k = int(k) if k is not None else self.k_features

        for _ in range(min(target_k, self.input_dim)):
            # 1) Train subnetwork on current selected set
            self._train_subnetwork(Xtr_t, ytr_t, Xval_t, yval_t)

            # 2) Score candidates via multi-dropout gradients
            j, _, _ = self._score_candidates(Xtr_t, ytr_t)

            # 3) Activate feature j
            self.selected_mask_[j] = True
            self.ranking_.append(j)

            # Xavier init for the new column
            fan_in = self.model.first.in_features
            fan_out = self.model.first.out_features
            bound = xavier_bound(fan_in, fan_out)
            with torch.no_grad():
                self.model.W1[:, j].uniform_(-bound, bound)

            self._zero_candidate_cols()

        return self

    @torch.no_grad()
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.scaler_ is not None:
            mu, sd = self.scaler_
            X = (X - mu) / sd
        Xt, _ = self._to_tensor(X, None)
        self.model.eval()
        logits = self.model(Xt)
        p = torch.sigmoid(logits).cpu().numpy()
        return np.vstack([1 - p, p]).T

    def selected_features_(self):
        return list(self.ranking_)


# -----------------------------
# CLI + main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Deep Neural Pursuit (DNP) — PyTorch")
    p.add_argument('--data', default='./data/data_0.csv', help='CSV: shape (n_samples, n_features)')
    p.add_argument('--features', default='./data/features_0.csv', help='CSV: feature names (n_features,)')
    p.add_argument('--labels', default='./data/labels.csv', help='CSV: labels (n_samples,)')

    p.add_argument('--k', type=int, default=7, help='Number of features to select')
    p.add_argument('--hidden', type=str, default='50,30,15', help='Hidden sizes, comma-separated')
    p.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    p.add_argument('--dropout-samples', type=int, default=12, help='Dropout passes for gradient averaging')
    p.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    p.add_argument('--epochs', type=int, default=1000, help='Max epochs per greedy iteration')
    p.add_argument('--patience', type=int, default=20, help='Early stopping patience per iteration')

    p.add_argument('--device', type=str, default="auto", help='cpu/cuda/auto')  # safer default
    p.add_argument('--seed', type=int, default=42, help='Random seed')

    p.add_argument('--standardize', action='store_true', help='Standardize features (z-score)')
    p.add_argument('--no-standardize', dest='standardize', action='store_false')
    p.set_defaults(standardize=True)

    p.add_argument('--stability-runs', type=int, default=0, help='Optional bootstrap runs for stability')
    p.add_argument('--output-prefix', type=str, default='selected_genes_dnp', help='Output prefix')
    return p.parse_args()


def safe_auc(y_true, y_score):
    if not SKLEARN_OK:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))  # correct union


def main():
    args = parse_args()

    # -----------------------------
    # 1. Load data
    # -----------------------------
    X = pd.read_csv(args.data, header=None).values.astype(np.float32)
    genes = pd.read_csv(args.features, header=None)[0].astype(str).tolist()
    y = pd.read_csv(args.labels, header=None)[0].values.astype(np.float32)

    assert X.shape[1] == len(genes), f"X has {X.shape[1]} cols, features file has {len(genes)} names"
    assert X.shape[0] == len(y), f"X has {X.shape[0]} rows, labels file has {len(y)} labels"

    # Convert to DataFrame for consistency with other scripts
    X_df = pd.DataFrame(X, columns=genes)

    hidden = tuple(int(x) for x in args.hidden.split(','))

    # -----------------------------
    # 2. Fit DNP
    # -----------------------------
    clf = DNPClassifier(
        input_dim=X.shape[1],
        hidden_layers=hidden,
        k_features=args.k,
        dropout_p=args.dropout,
        n_dropout_samples=args.dropout_samples,
        lr=args.lr,
        max_epochs_per_iter=args.epochs,
        early_stopping_patience=args.patience,
        device=args.device,
        seed=args.seed
    )

    clf.fit(X, y, standardize=args.standardize)

    # -----------------------------
    # 3. Extract selection
    # -----------------------------
    sel_idx = clf.selected_features_()
    sel_genes = [genes[j] for j in sel_idx]

    # -----------------------------
    # 4. Build feature ranking (FULL)
    # -----------------------------
    # You must have something like this in your classifier
    # e.g. gradient norms or importance scores
    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    elif hasattr(clf, "grad_norms_"):
        scores = clf.grad_norms_
    else:
        # fallback: zero importance except selected
        scores = np.zeros(len(genes), dtype=float)
        scores[sel_idx] = np.linspace(1.0, 0.5, len(sel_idx))

    importance_df = pd.DataFrame({
        "feature": genes,
        "importance": scores
    })

    importance_df = importance_df.sort_values(by="importance", ascending=False)

    # -----------------------------
    # 5. Reduce dataset
    # -----------------------------
    X_selected = X_df[sel_genes]

    print("Selected features:", len(sel_genes))
    print("Reduced shape:", X_selected.shape)

    # -----------------------------
    # 6. Save outputs (MATCH OTHERS)
    # -----------------------------
    X_selected.to_csv("X_dnp_selected.csv", index=False)
    importance_df.to_csv("dnp_feature_ranking.csv", index=False)
    pd.Series(sel_genes).to_csv(
        "dnp_selected_features.txt",
        index=False,
        header=False
    )

    print("\n✅ Files saved:")
    print(" - X_dnp_selected.csv")
    print(" - dnp_feature_ranking.csv")
    print(" - dnp_selected_features.txt")

    # -----------------------------
    # 7. Optional AUC (same as before)
    # -----------------------------
    proba_full = clf.predict_proba(X)[:, 1]
    auc_full = safe_auc(y, proba_full)

    if auc_full is not None:
        print(f"AUC (indicative, on full X): {auc_full:.4f}")

    # -----------------------------
    # 8. Optional stability (unchanged)
    # -----------------------------
    if args.stability_runs > 0:
        rng = np.random.RandomState(args.seed)
        selections = []

        for r in range(args.stability_runs):
            idx = rng.choice(np.arange(len(X)), size=len(X), replace=True)
            Xr, yr = X[idx], y[idx]

            dnp_r = DNPClassifier(
                input_dim=X.shape[1],
                hidden_layers=hidden,
                k_features=args.k,
                dropout_p=args.dropout,
                n_dropout_samples=args.dropout_samples,
                lr=args.lr,
                max_epochs_per_iter=args.epochs,
                early_stopping_patience=args.patience,
                device=args.device,
                seed=args.seed + r + 1
            )

            dnp_r.fit(Xr, yr, standardize=args.standardize)
            selections.append(dnp_r.selected_features_())

        jaccs = []
        for i in range(len(selections)):
            for j in range(i + 1, len(selections)):
                jaccs.append(jaccard(selections[i], selections[j]))

        stab_mean = float(np.mean(jaccs)) if jaccs else float('nan')

        with open('stability_report.txt', 'w', encoding='utf-8') as f:
            f.write("DNP Stability (Jaccard over selections)\n")
            f.write("=======================================\n")
            f.write(f"runs: {len(selections)}\n")
            f.write(f"mean_jaccard: {stab_mean:.4f}\n")

        print("Saved stability report: stability_report.txt")
        
if __name__ == '__main__':
    main()