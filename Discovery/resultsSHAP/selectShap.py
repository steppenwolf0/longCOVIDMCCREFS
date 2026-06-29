import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap

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
# 2. Train model
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42
)

rf.fit(X, y)

# -----------------------------
# 3. Compute SHAP values (SAFE)
# -----------------------------
explainer = shap.TreeExplainer(rf)
shap_vals = explainer.shap_values(X)

# ✅ Handle binary/multiclass cases
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

# ✅ Convert to numpy
shap_vals = np.array(shap_vals)

# ✅ Ensure 2D (samples × features)
if shap_vals.ndim == 3:
    shap_vals = shap_vals.mean(axis=-1)

print("SHAP shape:", shap_vals.shape)

# -----------------------------
# 4. Feature importance
# -----------------------------
shap_importance = np.abs(shap_vals).mean(axis=0)
shap_importance = shap_importance.flatten()

# -----------------------------
# 5. Create ranking
# -----------------------------
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": shap_importance
})

importance_df = importance_df.sort_values(by="importance", ascending=False)

# -----------------------------
# 6. Select top features
# -----------------------------
top_n = 7   # change this if needed
selected_features = importance_df["feature"].head(top_n)

print("Selected features:", len(selected_features))

# -----------------------------
# 7. Reduce dataset
# -----------------------------
X_selected = X[selected_features]

print("Reduced shape:", X_selected.shape)

# -----------------------------
# 8. Save outputs
# -----------------------------
X_selected.to_csv("X_shap_selected.csv", index=False)
importance_df.to_csv("shap_feature_ranking.csv", index=False)
pd.Series(selected_features).to_csv("shap_selected_features.txt", index=False, header=False)

print("\n✅ Files saved:")
print(" - X_shap_selected.csv")
print(" - shap_feature_ranking.csv")
print(" - shap_selected_features.txt")
