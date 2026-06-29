import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# -----------------------------
# 1. Load data
# -----------------------------
X = pd.read_csv("./data/data_0.csv", header=None)
features = pd.read_csv("./data/features_0.csv", header=None)[0].values
y = pd.read_csv("./data/labels.csv", header=None)[0].values

# Assign feature names
X.columns = features
print("Original shape:", X.shape)

# -----------------------------
# 2. Standardize (important)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Base model
# -----------------------------
model = LogisticRegression(
    penalty='l2',            # stable for RFE
    solver='liblinear',
    class_weight='balanced',
    max_iter=5000,
    random_state=42
)

# -----------------------------
# 4. RFECV (automatic selection)
# -----------------------------
rfecv = RFECV(
    estimator=model,
    step=1,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rfecv.fit(X_scaled, y)

print("Optimal number of features:", rfecv.n_features_)

# -----------------------------
# 5. Extract ranking
# -----------------------------
ranking = rfecv.ranking_

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": -ranking  # higher = better
}).sort_values(by="importance", ascending=False)

# -----------------------------
# 6. Selected features
# -----------------------------
selected_features = X.columns[rfecv.support_]

print("Selected features:", len(selected_features))

# -----------------------------
# 7. Reduce dataset
# -----------------------------
X_selected = X[selected_features]
print("Reduced shape:", X_selected.shape)

# -----------------------------
# 8. Save outputs
# -----------------------------
X_selected.to_csv("X_rfe_selected.csv", index=False)
importance_df.to_csv("rfe_feature_ranking.csv", index=False)
pd.Series(selected_features).to_csv(
    "rfe_selected_features.txt",
    index=False,
    header=False
)

print("\n✅ Files saved:")
print(" - X_rfe_selected.csv")
print(" - rfe_feature_ranking.csv")
print(" - rfe_selected_features.txt")