import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)
# -----------------------------
# 1. Load data
# -----------------------------
X = pd.read_csv("./data/data_0.csv", header=None)
features = pd.read_csv("./data/features_0.csv", header=None)[0].values
y = pd.read_csv("./data/labels.csv", header=None)[0].values

X.columns = features

print("Original shape:", X.shape)

# -----------------------------
# 2. Train Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=2000,       # large for stability
    n_jobs=-1,
    class_weight='balanced',
    max_depth=5,             # important for low sample size
    random_state=42
)

rf.fit(X, y)

# -----------------------------
# 3. Get feature importance
# -----------------------------
importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
})

# Sort descending
importance_df = importance_df.sort_values(by="importance", ascending=False)

# -----------------------------
# 4. Select top features
# -----------------------------
# OPTION A: top N features
top_n = 7
selected_features = importance_df["feature"].head(top_n)

# OPTION B: threshold-based (alternative)
# threshold = np.percentile(importances, 90)
# selected_features = importance_df[importance_df["importance"] > threshold]["feature"]

print("\nSelected features:", len(selected_features))

# -----------------------------
# 5. Reduce dataset
# -----------------------------
X_selected = X[selected_features]

print("Reduced shape:", X_selected.shape)

# -----------------------------
# 6. Save outputs
# -----------------------------
X_selected.to_csv("X_rf_selected.csv", index=False)

importance_df.to_csv("rf_feature_importance_ranking.csv", index=False)

pd.Series(selected_features).to_csv("rf_selected_features.txt", index=False, header=False)

print("\n Files saved:")
print(" - X_rf_selected.csv")
print(" - rf_feature_importance_ranking.csv")
print(" - rf_selected_features.txt")