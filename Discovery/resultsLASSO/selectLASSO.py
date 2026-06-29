import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

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
# 2. Standardize (MANDATORY)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. LASSO model (L1 only)
# -----------------------------
model = LogisticRegressionCV(
    penalty='l1',
    solver='saga',          # required for L1
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
coef = model.coef_.flatten()   # (n_features,)
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

# ✅ OPTION A (recommended): non-zero coefficients
selected_features = importance_df[importance_df.importance > 0]["feature"]

# ✅ OPTION B (stricter): top N
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
X_selected.to_csv("X_lasso_selected.csv", index=False)
importance_df.to_csv("lasso_feature_ranking.csv", index=False)
pd.Series(selected_features).to_csv("lasso_selected_features.txt", index=False, header=False)

print("\n✅ Files saved:")
print(" - X_lasso_selected.csv")
print(" - lasso_feature_ranking.csv")
print(" - lasso_selected_features.txt")