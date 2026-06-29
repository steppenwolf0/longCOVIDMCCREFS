import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# -----------------------------
# 1. Load data
# -----------------------------
def loadDataset():
    X = pd.read_csv("./data/data_0.csv", header=None)
    y = pd.read_csv("./data/labels.csv", header=None)[0].values
    features = pd.read_csv("./data/features_0.csv", header=None)[0].values

    X.columns = features
    return X, y, features


# -----------------------------
# 2. Run SelectKBest
# -----------------------------
def run_kbest(k=7):

    # Load
    X, y, features = loadDataset()

    print("Original shape:", X.shape)

    # -----------------------------
    # 3. Standardize (important)
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # 4. Fit SelectKBest
    # -----------------------------
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_scaled, y)

    scores = selector.scores_
    scores = np.nan_to_num(scores)

    # -----------------------------
    # 5. Build ranking (STANDARD FORMAT)
    # -----------------------------
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": scores
    })

    importance_df = importance_df.sort_values(by="importance", ascending=False)

    # -----------------------------
    # 6. Select features
    # -----------------------------
    selected_mask = selector.get_support()
    selected_features = features[selected_mask]

    print("Selected features:", len(selected_features))

    # -----------------------------
    # 7. Reduce dataset
    # -----------------------------
    X_selected = X[selected_features]

    print("Reduced shape:", X_selected.shape)

    # -----------------------------
    # 8. Save outputs (MATCH PIPELINE)
    # -----------------------------
    X_selected.to_csv("X_kbest_selected.csv", index=False)

    importance_df.to_csv(
        "kbest_feature_ranking.csv",
        index=False
    )

    pd.Series(selected_features).to_csv(
        "kbest_selected_features.txt",
        index=False,
        header=False
    )

    print("\n✅ Files saved:")
    print(" - X_kbest_selected.csv")
    print(" - kbest_feature_ranking.csv")
    print(" - kbest_selected_features.txt")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run_kbest(k=7)   # you can change k here