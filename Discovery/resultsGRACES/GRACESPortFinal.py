
# GRACESPort.py
# ------------------------------------------------------------
# Run GRACES feature selection from CSVs, but *without* changing GRACES.py.
# This wrapper performs iterative single-feature selection to reach k=7,
# adds preprocessing, stability selection across seeds, and evaluation.
# ------------------------------------------------------------

import os
import json
import random
import numpy as np
import pandas as pd

from GRACES import GRACES  # keep original GRACES implementation

# ---------------------------
# Config (tweak if needed)
# ---------------------------
K_TARGET = 7                 # final number of features to report
N_SEEDS = 25                 # stability selection seeds
N_DROPOUTS = 30              # gradient averaging inside GRACES
DROPOUT_PROB = 0.20          # lower dropout for small-k stability
EPOCHS = 150                 # a bit more training for convergence
LEARNING_RATE = 5e-4         # slightly smaller step
ALPHA = 0.90                 # sparser batch-wise graph
Q_NORM = 2                   # gradient norm p
SIGMA = 0.0                  # no noise in weights
F_CORRECT = 0.15             # blend in F-test a little

EVAL_ENABLE = True           # run CV evaluation of the final signature
EVAL_N_SPLITS = 5            # StratifiedKFold splits
EVAL_N_SEEDS = 5             # CV repeats with different splits

# ---------------------------
# Data loading (same format)
# ---------------------------
def load_data():
    """
    Reads CSV inputs exactly as in your current pipeline:
    - ./data/data_0.csv : (N_samples, N_genes) floats
    - ./data/features_0.csv : single column gene names
    - ./data/labels.csv : single column integer labels
    Returns:
      X_np (float32), gene_names (list[str]), y_np (int64)
    """
    data = pd.read_csv("./data/data_0.csv", header=None)
    features = pd.read_csv("./data/features_0.csv", header=None)
    labels = pd.read_csv("./data/labels.csv", header=None)

    X_np = data.values.astype(np.float32)      # shape (N_samples, N_genes)
    gene_names = features[0].tolist()          # single column with gene names
    y_np = labels.values.flatten().astype(np.int64)  # shape (N_samples,)
    return X_np, gene_names, y_np

# ---------------------------
# Preprocessing helpers
# ---------------------------
def zscore_per_feature(X):
    """Standardize each gene across samples, safe for constant columns."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    # Replace any residual NaNs/Infs
    Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)
    return Xz

def set_global_seed(seed):
    """Set seeds for reproducibility of NumPy and Python's random (GRACES uses NumPy)."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass  # torch might not be available or used here

# ---------------------------
# GRACES single-feature wrapper
# ---------------------------
def select_one_feature(X_np, y_np, params):
    """
    Run GRACES with n_features=1 on the *current* X_np and y_np.
    Returns: (index_in_current_X, success_flag)
    """
    try:
        gr = GRACES(
            n_features=1,
            hidden_size=None,     # falls back to [64, 32] per GRACES.py
            q=params["q"],
            n_dropouts=params["n_dropouts"],
            dropout_prob=params["dropout_prob"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            epochs=params["epochs"],
            alpha=params["alpha"],
            sigma=params["sigma"],
            f_correct=params["f_correct"],
        )
        sel_idx = gr.select(X_np, y_np)  # should return [int] with one selected feature
        if isinstance(sel_idx, (list, tuple)) and len(sel_idx) >= 1:
            return int(sel_idx[0]), True
        elif isinstance(sel_idx, np.ndarray) and sel_idx.size >= 1:
            return int(sel_idx.ravel()[0]), True
        else:
            return -1, False
    except Exception as e:
        print(f"[WARN] GRACES single-feature selection failed: {e}")
        return -1, False

# ---------------------------
# Iterative selection (k=7)
# ---------------------------
def iterative_k_selection(X_np, gene_names, y_np, k, params, seed=None):
    """
    Iteratively call GRACES(n_features=1), remove the selected column each round,
    and accumulate k features. Returns the *gene names* selected in order.
    """
    if seed is not None:
        set_global_seed(seed)

    X_work = X_np.copy()
    genes_work = gene_names.copy()
    selected_genes = []
    selected_ranks = []  # position when selected (1..k)

    for r in range(1, k + 1):
        idx, ok = select_one_feature(X_work, y_np, params)
        if not ok or idx < 0 or idx >= X_work.shape[1]:
            print(f"[WARN] Round {r}: selection failed or invalid index; stopping early.")
            break

        gene = genes_work[idx]
        selected_genes.append(gene)
        selected_ranks.append(r)

        # Remove the selected column so next rounds don't pick it again
        mask = np.ones(X_work.shape[1], dtype=bool)
        mask[idx] = False
        X_work = X_work[:, mask]
        genes_work = [g for i, g in enumerate(genes_work) if i != idx]

    return selected_genes, selected_ranks

# ---------------------------
# Stability selection
# ---------------------------
def stability_selection(X_np, gene_names, y_np, k, n_seeds, params):
    """
    Run iterative selection across multiple seeds, tally selection frequencies,
    and pick the top-k genes by frequency. Also compute mean selection rank.
    Returns:
      final_genes (list[str]), freq_df (DataFrame with freq & mean_rank)
    """
    all_picks = []
    all_orders = []  # list of dicts gene->rank (1..k)

    for s in range(n_seeds):
        seed = 1234 + s
        picks, ranks = iterative_k_selection(X_np, gene_names, y_np, k, params, seed)
        all_picks.append(picks)
        all_orders.append({g: r for g, r in zip(picks, ranks)})
        print(f"[INFO] Seed {seed}: selected {len(picks)} genes -> {picks}")

    # Tally frequencies and mean ranks
    freq = {}
    rank_sum = {}
    for order_dict in all_orders:
        for g, r in order_dict.items():
            freq[g] = freq.get(g, 0) + 1
            rank_sum[g] = rank_sum.get(g, 0) + r

    genes = sorted(freq.keys(), key=lambda g: (-freq[g], rank_sum[g]/freq[g]))
    final_genes = genes[:k]

    freq_df = pd.DataFrame({
        "gene": list(freq.keys()),
        "frequency": [freq[g] for g in freq.keys()],
        "mean_rank": [rank_sum[g] / freq[g] for g in freq.keys()],
    }).sort_values(by=["frequency", "mean_rank"], ascending=[False, True])

    return final_genes, freq_df

# ---------------------------
# Evaluation (optional)
# ---------------------------
def evaluate_signature(X_np, gene_names, y_np, sel_genes, n_splits=5, n_seeds=5):
    """
    Simple CV with Logistic Regression to sanity-check the 7-gene signature.
    Saves AUC (binary) and accuracy.
    """
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, accuracy_score

        # Map genes to indices
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        sel_idx = [gene_to_idx[g] for g in sel_genes if g in gene_to_idx]

        aucs, accs = [], []
        is_binary = (len(np.unique(y_np)) == 2)

        for seed in range(n_seeds):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for tr, te in skf.split(X_np, y_np):
                X_tr, X_te = X_np[tr][:, sel_idx], X_np[te][:, sel_idx]
                y_tr, y_te = y_np[tr], y_np[te]

                clf = LogisticRegression(penalty='l2', solver='liblinear', max_iter=300)
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)

                accs.append(accuracy_score(y_te, y_pred))
                if is_binary:
                    y_proba = clf.predict_proba(X_te)[:, 1]
                    aucs.append(roc_auc_score(y_te, y_proba))

        metrics = {
            "accuracy_mean": float(np.mean(accs)) if len(accs) else None,
            "accuracy_std": float(np.std(accs)) if len(accs) else None,
        }
        if is_binary and len(aucs):
            metrics.update({
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
            })
        return metrics
    except Exception as e:
        print(f"[WARN] Evaluation skipped: {e}")
        return {"error": str(e)}


def run_graces_feature_selection(
    n_features=K_TARGET,
    dropout_prob=DROPOUT_PROB,
    f_correct=F_CORRECT,
    n_dropouts=N_DROPOUTS,
    hidden_size=None,
    q=Q_NORM,
    batch_size=16,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    alpha=ALPHA,
    sigma=SIGMA
):
    """
    Standardized GRACES pipeline with robust output formatting
    """

    os.makedirs("./best", exist_ok=True)

    # -----------------------------
    # 1. Load data
    # -----------------------------
    X_np, gene_names, y_np = load_data()

    # -----------------------------
    # 2. Preprocess
    # -----------------------------
    X_np = zscore_per_feature(X_np)

    # -----------------------------
    # 3. Stability selection
    # -----------------------------
    params = dict(
        q=q,
        n_dropouts=n_dropouts,
        dropout_prob=dropout_prob,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        alpha=alpha,
        sigma=sigma,
        f_correct=f_correct,
    )

    final_genes, freq_df = stability_selection(
        X_np=X_np,
        gene_names=gene_names,
        y_np=y_np,
        k=n_features,
        n_seeds=N_SEEDS,
        params=params
    )

    print("Selected features:", len(final_genes))

    # -----------------------------
    # 4. Build importance ranking (ROBUST FIX)
    # -----------------------------
    freq_df = freq_df.copy()

    # 🔍 Detect feature column automatically
    if "feature" in freq_df.columns:
        feature_col = "feature"
    elif "gene" in freq_df.columns:
        feature_col = "gene"
    else:
        # fallback: assume first column is feature names
        feature_col = freq_df.columns[0]

    # Avoid division by zero
    if "mean_rank" in freq_df.columns:
        freq_df["mean_rank"] = freq_df["mean_rank"].replace(0, 1e-6)
    else:
        freq_df["mean_rank"] = 1.0  # fallback

    if "frequency" not in freq_df.columns:
        raise ValueError("freq_df must contain a 'frequency' column")

    # Importance formula
    freq_df["importance"] = freq_df["frequency"] / freq_df["mean_rank"]

    # ✅ Standardized ranking format (MATCH SHAP / EN)
    importance_df = pd.DataFrame({
        "feature": freq_df[feature_col],
        "importance": freq_df["importance"]
    })

    importance_df = importance_df.sort_values(by="importance", ascending=False)

    # -----------------------------
    # 5. Reduce dataset
    # -----------------------------
    X_df = pd.DataFrame(X_np, columns=gene_names)
    X_selected = X_df[final_genes]

    print("Reduced shape:", X_selected.shape)

    # -----------------------------
    # 6. Save outputs (STANDARD)
    # -----------------------------
    X_selected.to_csv("X_graces_selected.csv", index=False)

    importance_df.to_csv(
        "graces_feature_ranking.csv",
        index=False
    )

    pd.Series(final_genes).to_csv(
        "graces_selected_features.txt",
        index=False,
        header=False
    )

    print("\n✅ Files saved (standardized):")
    print(" - X_graces_selected.csv")
    print(" - graces_feature_ranking.csv")
    print(" - graces_selected_features.txt")

    # -----------------------------
    # 7. Legacy outputs (optional)
    # -----------------------------
    pd.DataFrame({"features": [";".join(final_genes)]}).to_csv(
        "./best/signature_graces.csv",
        index=False
    )

    pd.DataFrame(final_genes).to_csv(
        "./best/features_0.csv",
        header=None,
        index=None
    )

    freq_df.to_csv("./best/selection_freq.csv", index=False)

    # -----------------------------
    # 8. Optional evaluation
    # -----------------------------
    if EVAL_ENABLE:
        metrics = evaluate_signature(
            X_np, gene_names, y_np, final_genes,
            n_splits=EVAL_N_SPLITS,
            n_seeds=EVAL_N_SEEDS
        )

        with open("./best/eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[CV] Metrics saved to ./best/eval_metrics.json -> {metrics}")

if __name__ == "__main__":
    run_graces_feature_selection(n_features=K_TARGET)

