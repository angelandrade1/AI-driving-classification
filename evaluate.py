"""
evaluate.py — Model Evaluation

Loads every trained model from models/ and evaluates it on the same D6 test
split used in train.py. Generates a confusion matrix heatmap for each model
and writes the final comparison table.

  figures/confusion_matrix_<model>.png  — one heatmap per model (7 total)
  results/model_comparison.csv          — accuracy, precision, recall, F1
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "processed_dataset.csv")
MODELS_DIR  = os.path.join(SCRIPT_DIR, "models")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(RESULTS_DIR, "model_comparison.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_NAMES  = {0: "Normal", 1: "Conservative", 2: "Spirited"}
TARGET_NAMES = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]

FEATURE_COLS = [
    "speed_std", "speed_range", "speed_cv", "speed_max_ratio",
    "accel_x_std", "accel_x_min", "norm_braking",
    "accel_y_std", "accel_y_max", "norm_lateral",
    "accel_magnitude_mean", "accel_magnitude_std",
    "lane_deviation_mean",
    "road_type",
]

# Slug (display name, file path)
MODEL_REGISTRY = {
    "random_forest": ("Random Forest", os.path.join(MODELS_DIR, "random_forest.pkl")),
    "knn":           ("KNN",           os.path.join(MODELS_DIR, "knn.pkl")),
    "gaussian_nb":   ("Gaussian NB",   os.path.join(MODELS_DIR, "gaussian_nb.pkl")),
    "svm":           ("SVM",           os.path.join(MODELS_DIR, "svm.pkl")),
    "xgboost":       ("XGBoost",       os.path.join(MODELS_DIR, "xgboost.pkl")),
    "mlp":           ("MLP",           os.path.join(MODELS_DIR, "mlp.pkl")),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_section(title: str):
    bar = "=" * 62
    print(f"\n{bar}\n  {title}\n{bar}")


def load_model(display_name: str, path: str):
    """Load a model from disk; return None and warn if missing."""
    if not os.path.isfile(path):
        print(f"  [WARN] Model file not found: {path} — skipping {display_name}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save_confusion_matrix(y_true, y_pred, display_name: str, slug: str):
    """Plot and save a seaborn heatmap for the confusion matrix."""
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label",      fontsize=11)
    ax.set_title(f"Confusion Matrix — {display_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{slug}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Heatmap saved → {out_path}")


def collect_metrics(display_name: str, y_true, y_pred) -> dict:
    """Return a flat metrics dict for the comparison table."""
    acc = accuracy_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_cls, r_cls, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "Model":                   display_name,
        "Accuracy":                round(acc,      4),
        "Precision_weighted":      round(p_w,      4),
        "Recall_weighted":         round(r_w,      4),
        "F1_weighted":             round(f1_w,     4),
        "Precision_Normal":        round(p_cls[0], 4),
        "Recall_Normal":           round(r_cls[0], 4),
        "Precision_Conservative":  round(p_cls[1], 4),
        "Recall_Conservative":     round(r_cls[1], 4),
        "Precision_Spirited":      round(p_cls[2], 4),
        "Recall_Spirited":         round(r_cls[2], 4),
    }


# ---------------------------------------------------------------------------
# 1. Reconstruct the same D6 test split used during training
# ---------------------------------------------------------------------------
print_section("Loading data & reconstructing test split")

df = pd.read_csv(DATA_PATH)

# Same engineered features as train.py — has to match exactly or the model sees different inputs
df["speed_cv"]        = df["speed_std"]    / (df["speed_mean"] + 1.0)
df["speed_max_ratio"] = df["speed_max"]    / (df["speed_mean"] + 1.0)
df["norm_braking"]    = -df["accel_x_min"] / (df["speed_mean"] + 1.0)
df["norm_lateral"]    = df["accel_y_max"]  / (df["speed_mean"] + 1.0)

# Per-driver z-score normalization — same as train.py, applied before the scaler
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

X = feat_arr
y = df["label"].values

if "driver" in df.columns:
    print("  Using driver-based split  (D1–D5 train | D6 test)")
    mask = df["driver"].isin(["D1", "D2", "D3", "D4", "D5"]).values
    X_train_raw, y_train = X[mask], y[mask]
    X_test_raw,  y_test  = X[~mask], y[~mask]
else:
    print("  'driver' column not found — using 80/20 stratified random split")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

# Apply the same scaler that was fit during training
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

X_train_sc = scaler.transform(X_train_raw)
X_test_sc  = scaler.transform(X_test_raw)

print(f"  Test set  : {len(X_test_sc):,} windows")
for code, name in LABEL_NAMES.items():
    print(f"    {name:<14}: {int((y_test == code).sum())}")

# ---------------------------------------------------------------------------
# 2. Evaluate each model
# ---------------------------------------------------------------------------
all_metrics: list[dict] = []

loaded_models: dict = {}

for slug, (display_name, path) in MODEL_REGISTRY.items():
    print_section(f"Evaluating — {display_name}")

    model = load_model(display_name, path)
    if model is None:
        continue

    loaded_models[slug] = model
    y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall accuracy : {acc:.4f}  ({acc*100:.2f} %)")

    print("\n  Classification report:")
    print(classification_report(
        y_test, y_pred, target_names=TARGET_NAMES, digits=4
    ))

    # Confusion matrix — text
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    header = "            " + "  ".join(f"{n:>14}" for n in TARGET_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {TARGET_NAMES[i]:>10}  " + "  ".join(f"{v:>14}" for v in row))

    # Confusion matrix heatmap
    save_confusion_matrix(y_test, y_pred, display_name, slug)

    all_metrics.append(collect_metrics(display_name, y_test, y_pred))

# ---------------------------------------------------------------------------
# 2b. Soft Voting Ensemble — same SVM + MLP average as in train.py
# ---------------------------------------------------------------------------
if "svm" in loaded_models and "mlp" in loaded_models:
    print_section("Evaluating — Soft Voting Ensemble (SVM + MLP)")

    avg_proba = (
        loaded_models["svm"].predict_proba(X_test_sc)
        + loaded_models["mlp"].predict_proba(X_test_sc)
    ) / 2.0
    y_pred_ensemble = np.argmax(avg_proba, axis=1)

    acc = accuracy_score(y_test, y_pred_ensemble)
    print(f"\n  Overall accuracy : {acc:.4f}  ({acc*100:.2f} %)")

    print("\n  Classification report:")
    print(classification_report(
        y_test, y_pred_ensemble, target_names=TARGET_NAMES, digits=4
    ))

    cm = confusion_matrix(y_test, y_pred_ensemble)
    print("  Confusion matrix (rows=actual, cols=predicted):")
    header = "            " + "  ".join(f"{n:>14}" for n in TARGET_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {TARGET_NAMES[i]:>10}  " + "  ".join(f"{v:>14}" for v in row))

    save_confusion_matrix(y_test, y_pred_ensemble, "Soft Voting Ensemble", "soft_voting_ensemble")
    all_metrics.append(collect_metrics("Soft Voting Ensemble", y_test, y_pred_ensemble))
else:
    print("\n  [WARN] SVM or MLP model not loaded — skipping Soft Voting Ensemble")

# ---------------------------------------------------------------------------
# 3. Save comparison table
# ---------------------------------------------------------------------------
print_section("Model Comparison — All Results")

results_df = (
    pd.DataFrame(all_metrics)
    .sort_values("Accuracy", ascending=False)
    .reset_index(drop=True)
)

display_cols = ["Model", "Accuracy", "Precision_weighted", "Recall_weighted", "F1_weighted"]
print(results_df[display_cols].to_string(index=False))

results_df.to_csv(RESULTS_CSV, index=False)
print(f"\n  Comparison table saved → {RESULTS_CSV}")

# ---------------------------------------------------------------------------
# 4. Summary of saved figures
# ---------------------------------------------------------------------------
print_section("Saved figures")
all_slugs = list(MODEL_REGISTRY.keys()) + ["soft_voting_ensemble"]
for slug in all_slugs:
    p = os.path.join(FIGURES_DIR, f"confusion_matrix_{slug}.png")
    exists = "✓" if os.path.isfile(p) else "✗"
    print(f"  {exists}  {p}")
print()
