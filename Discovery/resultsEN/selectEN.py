import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
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
# 2. Standardize features (CRITICAL for Elastic Net)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Elastic Net model (with CV)
# -----------------------------
model = LogisticRegressionCV(
    penalty='elasticnet',
    solver='saga',        # required for elastic net
    l1_ratios=[0.1, 0.5, 0.7, 0.9],  # mix L1/L2
    Cs=10,
    cv=5,
    n_jobs=-1,
    max_iter=5000,
    class_weight='balanced',
    random_state=42
)

model.fit(X_scaled, y)

# -----------------------------
# 4. Extract coefficients
# -----------------------------
coef = model.coef_.flatten()   # shape: (n_features,)

# Use absolute value for importance
importance = np.abs(coef)

# -----------------------------
# 5. Create ranking
# -----------------------------
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

# -----------------------------
# 6. Select features
# -----------------------------

# OPTION A: select non-zero (classic Elastic Net)
selected_features = importance_df[importance_df.importance > 0]["feature"]

# OPTION B: top N (more aggressive)
# top_n = 30
# selected_features = importance_df["feature"].head(top_n)

print("Selected features:", len(selected_features))

# -----------------------------
# 7. Reduce dataset
# -----------------------------
X_selected = X[selected_features]

print("Reduced shape:", X_selected.shape)

# -----------------------------
# 8. Save outputs
# -----------------------------
X_selected.to_csv("X_enet_selected.csv", index=False)
importance_df.to_csv("enet_feature_ranking.csv", index=False)
pd.Series(selected_features).to_csv("enet_selected_features.txt", index=False, header=False)

print("\n Files saved:")
print(" - X_enet_selected.csv")
print(" - enet_feature_ranking.csv")
print(" - enet_selected_features.txt")
