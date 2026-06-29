import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
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
# 2. Define model (robust for HDLSS)
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=1000,      # IMPORTANT for stability
    n_jobs=-1,
    class_weight='balanced',
    max_depth=5,            # helps avoid overfitting in low n
    random_state=42
)

# -----------------------------
# 3. Run Boruta
# -----------------------------
boruta = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    max_iter=200,           # more iterations = more stable
    verbose=2,
    random_state=42
)

boruta.fit(X.values, y)

# -----------------------------
# 4. Extract selected features
# -----------------------------
selected_mask = boruta.support_
tentative_mask = boruta.support_weak_

selected_features = X.columns[selected_mask]
tentative_features = X.columns[tentative_mask]

print("\nConfirmed features:", len(selected_features))
print("Tentative features:", len(tentative_features))

# -----------------------------
# 5. Reduce dataset
# -----------------------------
X_selected = X[selected_features]

# Optionally include tentative features:
# X_selected = X[selected_features.union(tentative_features)]

print("Reduced shape:", X_selected.shape)

# -----------------------------
# 6. Save outputs
# -----------------------------
X_selected.to_csv("X_boruta_selected.csv", index=False)

pd.Series(selected_features).to_csv("selected_features.txt", index=False, header=False)
pd.Series(tentative_features).to_csv("tentative_features.txt", index=False, header=False)

print("\n Files saved:")
print(" - X_boruta_selected.csv (reduced feature matrix)")
print(" - selected_features.txt")
print(" - tentative_features.txt")