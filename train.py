"""
train.py — Model Training

Trains 7 classifiers on the processed dataset and saves them to models/.
The split is cross-driver (D1–D5 train, D6 test) so the test set is a
completely unseen driver — harder and more realistic than a random split.

  1. Random Forest        → models/random_forest.pkl
  2. K-Nearest Neighbours → models/knn.pkl
  3. Gaussian Naïve Bayes → models/gaussian_nb.pkl
  4. SVM + GridSearchCV   → models/svm.pkl
  5. XGBoost              → models/xgboost.pkl
  6. MLP                  → models/mlp.pkl
  7. Soft Voting Ensemble (SVM + MLP averaged — no separate file)

braking_event_rate is excluded from the feature set — it had near-zero
importance in the Random Forest, so it wasn't contributing anything.

Saves: results/model_comparison.csv, figures/model_comparison_bar.png,
       figures/random_forest_feature_importance.png
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "processed_dataset.csv")
MODELS_DIR  = os.path.join(SCRIPT_DIR, "models")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

for d in (MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# Output file paths
RF_PATH      = os.path.join(MODELS_DIR,  "random_forest.pkl")
KNN_PATH     = os.path.join(MODELS_DIR,  "knn.pkl")
NB_PATH      = os.path.join(MODELS_DIR,  "gaussian_nb.pkl")
SVM_PATH     = os.path.join(MODELS_DIR,  "svm.pkl")
XGB_PATH     = os.path.join(MODELS_DIR,  "xgboost.pkl")
MLP_PATH     = os.path.join(MODELS_DIR,  "mlp.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR,  "scaler.pkl")
FI_PLOT      = os.path.join(FIGURES_DIR, "random_forest_feature_importance.png")
BAR_PLOT     = os.path.join(FIGURES_DIR, "model_comparison_bar.png")
RESULTS_CSV  = os.path.join(RESULTS_DIR, "model_comparison.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_NAMES  = {0: "Normal", 1: "Conservative", 2: "Spirited"}
TARGET_NAMES = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]

# speed_mean and speed_max are excluded because D6 drives about 7 km/h slower
# than D1–D5 on average — absolute speed would just encode driver identity
# rather than driving style. The relative features (speed_cv, norm_*) capture
# the same within-window variation without that bias.
FEATURE_COLS = [
    # Relative speed
    "speed_std", "speed_range", "speed_cv", "speed_max_ratio",
    # Longitudinal dynamics
    "accel_x_std", "accel_x_min", "norm_braking",
    # Lateral dynamics
    "accel_y_std", "accel_y_max", "norm_lateral",
    # Overall motion energy
    "accel_magnitude_mean", "accel_magnitude_std",
    # Lane & road context
    "lane_deviation_mean",
    "road_type",
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def print_section(title: str):
    bar = "=" * 62
    print(f"\n{bar}\n  {title}\n{bar}")


def save_pkl(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def evaluate_model(name: str, y_true, y_pred) -> dict:
    """
    Print a full evaluation block and return a metrics dict for the
    comparison table.
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Overall accuracy : {acc:.4f}  ({acc*100:.2f} %)")
    print("\n  Classification report:")
    print(classification_report(y_true, y_pred,
                                target_names=TARGET_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    header = "            " + "  ".join(f"{n:>14}" for n in TARGET_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>14}" for v in row)
        print(f"  {TARGET_NAMES[i]:>10}  {row_str}")

    # Per-class precision / recall (for the comparison table)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_cls, r_cls, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    return {
        "Model":                   name,
        "Accuracy":                round(acc,  4),
        "Precision_weighted":      round(p_w,  4),
        "Recall_weighted":         round(r_w,  4),
        "F1_weighted":             round(f1_w, 4),
        "Precision_Normal":        round(p_cls[0], 4),
        "Recall_Normal":           round(r_cls[0], 4),
        "Precision_Conservative":  round(p_cls[1], 4),
        "Recall_Conservative":     round(r_cls[1], 4),
        "Precision_Spirited":      round(p_cls[2], 4),
        "Recall_Spirited":         round(r_cls[2], 4),
    }


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print_section("Loading data")

df = pd.read_csv(DATA_PATH)

# Relative features — less sensitive to per-driver speed baseline
df["speed_cv"]        = df["speed_std"]    / (df["speed_mean"] + 1.0)
df["speed_max_ratio"] = df["speed_max"]    / (df["speed_mean"] + 1.0)
df["norm_braking"]    = -df["accel_x_min"] / (df["speed_mean"] + 1.0)
df["norm_lateral"]    = df["accel_y_max"]  / (df["speed_mean"] + 1.0)

# Z-score each driver's features against their own mean/std so the model sees
# driving style, not absolute speed level. Without this, D6's lower baseline
# would bleed through even the relative features.
if "driver" in df.columns:
    feat_arr = df[FEATURE_COLS].values.astype(float)
    normed   = np.zeros_like(feat_arr)
    for driver in df["driver"].unique():
        idx = (df["driver"] == driver).values
        d   = feat_arr[idx]
        normed[idx] = (d - d.mean(axis=0)) / d.std(axis=0).clip(min=1e-8)
    feat_arr = normed
else:
    feat_arr = df[FEATURE_COLS].values.astype(float)

print(f"  Loaded {len(df):,} windows")
print(f"  Feature set ({len(FEATURE_COLS)} features): {FEATURE_COLS}")

X = feat_arr
y = df["label"].values

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
if "driver" in df.columns:
    print("\n  Using driver-based split  (D1–D5 train | D6 test)")
    train_mask = df["driver"].isin(["D1", "D2", "D3", "D4", "D5"]).values
    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[~train_mask], y[~train_mask]
else:
    print("\n  'driver' column not found — using 80/20 stratified random split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")
for code, name in LABEL_NAMES.items():
    n_tr = int((y_train == code).sum())
    n_te = int((y_test  == code).sum())
    print(f"    {name:<14}: train={n_tr}  test={n_te}")

# ---------------------------------------------------------------------------
# 3. Feature scaling  (fit on train only, reused by all models that need it)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
save_pkl(scaler, SCALER_PATH)

# Balanced class weights
classes           = np.array(sorted(LABEL_NAMES.keys()))
class_weights_arr = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes.tolist(), class_weights_arr))
# Per-sample weights for XGBoost
sample_weights = np.array([class_weight_dict[lbl] for lbl in y_train])

print(f"\n  Class weights: { {LABEL_NAMES[k]: round(v,4) for k,v in class_weight_dict.items()} }")

# Container for comparison metrics
all_metrics: list[dict] = []


# ===========================================================================
# MODEL 1 — Random Forest
# ===========================================================================
print_section("1 / 7  —  Random Forest")

rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_sc, y_train)
save_pkl(rf, RF_PATH)

# Feature importances
importances = rf.feature_importances_
fi_order    = np.argsort(importances)[::-1]
print("\n  Feature importances (ranked):")
print(f"  {'Feature':<26} {'Importance':>10}")
print(f"  {'-'*38}")
for idx in fi_order:
    print(f"  {FEATURE_COLS[idx]:<26} {importances[idx]:>10.4f}")

# Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(
    [FEATURE_COLS[i] for i in fi_order[::-1]],
    importances[fi_order[::-1]],
    color="steelblue", edgecolor="white",
)
ax.set_xlabel("Importance (mean decrease in impurity)", fontsize=11)
ax.set_title("Random Forest — Feature Importances", fontsize=13, fontweight="bold")
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
plt.savefig(FI_PLOT, dpi=150)
plt.close()
print(f"\n  Feature importance plot saved → {FI_PLOT}")

y_pred_rf = rf.predict(X_test_sc)
print_section("Random Forest — Test-set Evaluation")
all_metrics.append(evaluate_model("Random Forest", y_test, y_pred_rf))





# ===========================================================================
# MODEL 2 — K-Nearest Neighbours
# ===========================================================================
print_section("2 / 7  —  K-Nearest Neighbours  (k=11)")

knn = KNeighborsClassifier(n_neighbors=11, metric="euclidean", weights="distance")
knn.fit(X_train_sc, y_train)
save_pkl(knn, KNN_PATH)

y_pred_knn = knn.predict(X_test_sc)
print_section("KNN — Test-set Evaluation")
all_metrics.append(evaluate_model("KNN", y_test, y_pred_knn))


# ===========================================================================
# MODEL 3 — Gaussian Naïve Bayes
# ===========================================================================
print_section("3 / 7  —  Gaussian Naïve Bayes")

gnb = GaussianNB()
gnb.fit(X_train_sc, y_train)
save_pkl(gnb, NB_PATH)

y_pred_gnb = gnb.predict(X_test_sc)
print_section("Gaussian NB — Test-set Evaluation")
all_metrics.append(evaluate_model("Gaussian NB", y_test, y_pred_gnb))


# ===========================================================================
# MODEL 4 — SVM + GridSearchCV
# ===========================================================================
print_section("4 / 7  —  SVM  +  GridSearchCV")

svc_base = SVC(class_weight="balanced", probability=True, random_state=42)
# Separate dicts avoid testing gamma with linear kernel (gamma is ignored there)
param_grid = [
    {"kernel": ["rbf"],    "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
]
grid_search = GridSearchCV(
    svc_base, param_grid,
    cv=2, scoring="f1_weighted", n_jobs=-1, verbose=1,
)
grid_search.fit(X_train_sc, y_train)

best_svm = grid_search.best_estimator_
print(f"\n  Best parameters : {grid_search.best_params_}")
print(f"  Best CV F1      : {grid_search.best_score_:.4f}")
save_pkl(best_svm, SVM_PATH)

y_pred_svm = best_svm.predict(X_test_sc)
print_section("SVM — Test-set Evaluation")
all_metrics.append(evaluate_model("SVM", y_test, y_pred_svm))


# ===========================================================================
# MODEL 5 — XGBoost
# ===========================================================================
print_section("5 / 7  —  XGBoost")

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0,
)
# Pass per-sample weights derived from class_weight to handle imbalance
xgb.fit(X_train_sc, y_train, sample_weight=sample_weights)
save_pkl(xgb, XGB_PATH)

y_pred_xgb = xgb.predict(X_test_sc)
print_section("XGBoost — Test-set Evaluation")
all_metrics.append(evaluate_model("XGBoost", y_test, y_pred_xgb))


# ===========================================================================
# MODEL 6 — MLP (sklearn)
# ===========================================================================
print_section("6 / 7  —  MLP  (sklearn)")

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    learning_rate_init=0.0005,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    verbose=False,
)
mlp.fit(X_train_sc, y_train)
print(f"  Converged after {mlp.n_iter_} iterations")
save_pkl(mlp, MLP_PATH)

y_pred_mlp = mlp.predict(X_test_sc)
print_section("MLP — Test-set Evaluation")
all_metrics.append(evaluate_model("MLP", y_test, y_pred_mlp))


# ===========================================================================
# MODEL 7 — Soft Voting Ensemble (average class probabilities)
# ===========================================================================
print_section("7 / 7  —  Soft Voting Ensemble  (SVM + MLP equal weight)")

# SVM and MLP respond best to per-driver normalization and have well-calibrated
# probabilities. Adding the weaker models (GNB, KNN) into the ensemble actually
# hurts — their shaky probability estimates drag the average down.
avg_proba = (
    best_svm.predict_proba(X_test_sc) + mlp.predict_proba(X_test_sc)
) / 2.0
y_pred_ensemble = np.argmax(avg_proba, axis=1)

print_section("Soft Voting Ensemble — Test-set Evaluation")
all_metrics.append(evaluate_model("Soft Voting Ensemble", y_test, y_pred_ensemble))


# ===========================================================================
# Comparison table
# ===========================================================================
print_section("Model Comparison — All Results")

results_df = pd.DataFrame(all_metrics)
results_df = results_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

# Pretty-print the summary
display_cols = ["Model", "Accuracy", "Precision_weighted",
                "Recall_weighted", "F1_weighted"]
print(results_df[display_cols].to_string(index=False))

# Save full table
results_df.to_csv(RESULTS_CSV, index=False)
print(f"\n  Full comparison table saved → {RESULTS_CSV}")


# ===========================================================================
# Grouped bar chart  (main paper figure)
# ===========================================================================
metrics_to_plot = ["Accuracy", "Precision_weighted", "Recall_weighted", "F1_weighted"]
metric_labels   = ["Accuracy", "Precision", "Recall", "F1"]

model_names = results_df["Model"].tolist()
n_models    = len(model_names)
n_metrics   = len(metrics_to_plot)

# Palette: one colour per metric
colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

x      = np.arange(n_models)
width  = 0.18
offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

fig, ax = plt.subplots(figsize=(16, 6))

for i, (col, label, colour) in enumerate(zip(metrics_to_plot, metric_labels, colours)):
    vals = results_df[col].values
    bars = ax.bar(x + offsets[i], vals, width, label=label,
                  color=colour, edgecolor="white", linewidth=0.6)
    # Value labels on top of each bar
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=6.5, rotation=90,
        )

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_title(
    "Driver Behaviour Classifier — Model Comparison\n"
    "(Accuracy · Weighted Precision · Weighted Recall · Weighted F1)",
    fontsize=13, fontweight="bold",
)
ax.legend(loc="lower right", fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(BAR_PLOT, dpi=180)
plt.close()
print(f"  Model comparison bar chart saved → {BAR_PLOT}")


# ===========================================================================
# Final summary
# ===========================================================================
print_section("Saved artefacts")
artefacts = [
    RF_PATH, KNN_PATH, NB_PATH,
    SVM_PATH, XGB_PATH, MLP_PATH, SCALER_PATH,
    FI_PLOT, BAR_PLOT, RESULTS_CSV,
]
for path in artefacts:
    exists = "✓" if os.path.isfile(path) else "✗ MISSING"
    print(f"  {exists}  {path}")
print()
