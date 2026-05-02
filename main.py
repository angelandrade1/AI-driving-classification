"""
main.py — End-to-End Demo

Ties everything together. Grabs a real window from the held-out D6 test set,
runs it through the MLP classifier, and shows the personalized route
recommendation for the predicted driver profile.

Steps:
  1. Load processed_dataset.csv and reconstruct the D6 test split
  2. Find the first test window the model classifies correctly
  3. Classify it → Normal, Conservative, or Spirited
  4. Build the road graph and compute the profile's recommended route
  5. Print a summary and save the route visualization

Full pipeline from sensor data to personalized route in one script.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from routing import build_graph, get_routes, print_route, plot_routes, ROUTE_FIGURE

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "processed_dataset.csv")
MLP_PATH    = os.path.join(SCRIPT_DIR, "models", "mlp.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "models", "scaler.pkl")

# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "speed_std", "speed_range", "speed_cv", "speed_max_ratio",
    "accel_x_std", "accel_x_min", "norm_braking",
    "accel_y_std", "accel_y_max", "norm_lateral",
    "accel_magnitude_mean", "accel_magnitude_std",
    "lane_deviation_mean",
    "road_type",
]

LABEL_NAMES = {0: "Normal", 1: "Conservative", 2: "Spirited"}

# ---------------------------------------------------------------------------
# Profile descriptions printed to the user
# ---------------------------------------------------------------------------
PROFILE_DESCRIPTIONS = {
    "Normal": (
        "Standard route selected. "
        "Optimised for balanced travel time using all road types."
    ),
    "Conservative": (
        "Calm driving profile detected. "
        "Route avoids highways and favours quieter surface roads."
    ),
    "Spirited": (
        "Energetic driving profile detected. "
        "Route exploits shortcuts and uses highways at full efficiency."
    ),
}


# ---------------------------------------------------------------------------
# Step 1 — Load data and reconstruct the test split
# ---------------------------------------------------------------------------
def load_test_sample() -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X_test, y_test) using the same driver-based split as train.py so
    all samples are truly held-out data.
    """
    df = pd.read_csv(DATA_PATH)
    df["speed_cv"]        = df["speed_std"]    / (df["speed_mean"] + 1.0)
    df["speed_max_ratio"] = df["speed_max"]    / (df["speed_mean"] + 1.0)
    df["norm_braking"]    = -df["accel_x_min"] / (df["speed_mean"] + 1.0)
    df["norm_lateral"]    = df["accel_y_max"]  / (df["speed_mean"] + 1.0)
    # Per-driver z-score normalisation (mirrors train.py preprocessing)
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
    X  = feat_arr
    y  = df["label"].values

    if "driver" in df.columns:
        mask = df["driver"].isin(["D1", "D2", "D3", "D4", "D5"])
        X_test = X[~mask]
        y_test = y[~mask]
    else:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

    # Return all test windows so the caller can pick a correctly-classified one
    return X_test, y_test


# ---------------------------------------------------------------------------
# Step 2 — Classify
# ---------------------------------------------------------------------------
def classify_sample(feature_vector: np.ndarray) -> str:
    """
    Load the MLP and scaler, scale the sample, return profile label.
    """
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(MLP_PATH, "rb") as f:
        model = pickle.load(f)

    X_sc   = scaler.transform(feature_vector.reshape(1, -1))
    pred   = model.predict(X_sc)[0]
    proba  = model.predict_proba(X_sc)[0]

    profile = LABEL_NAMES[int(pred)]
    return profile, proba


# ---------------------------------------------------------------------------
# Step 3 — Route recommendation
# ---------------------------------------------------------------------------
def recommend_route(profile: str, origin: str = "Home", destination: str = "OfficeE"):
    """Build the graph, compute all three routes, return the profile's route."""
    G      = build_graph()
    routes = get_routes(G, origin, destination)
    return G, routes, routes[profile]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 62

    print(f"\n{sep}")
    print("  GPS Route Personalisation — End-to-End Demo")
    print(f"{sep}\n")

    # ------------------------------------------------------------------
    # 1. Load sample — pick first correctly-classified window in test set
    # ------------------------------------------------------------------
    print("  [1/3]  Loading demo sample from test set …")
    X_test, y_test = load_test_sample()

    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)
    with open(MLP_PATH, "rb") as f:
        _model = pickle.load(f)

    feature_vector, true_label = None, None
    for xv, yl in zip(X_test, y_test):
        pred = int(_model.predict(_scaler.transform(xv.reshape(1, -1)))[0])
        if pred == int(yl):
            feature_vector, true_label = xv, int(yl)
            break

    if feature_vector is None:
        # Fallback: no correct prediction found, use first sample anyway
        feature_vector, true_label = X_test[0], int(y_test[0])

    true_profile = LABEL_NAMES[true_label]

    print(f"\n  Feature vector (first correctly-classified window of test set):")
    for name, val in zip(FEATURE_COLS, feature_vector):
        print(f"    {name:<26} {val:.5f}")
    print(f"\n  Ground-truth label : {true_profile}  (encoded {true_label})")

    # ------------------------------------------------------------------
    # 2. Classify
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("  [2/3]  Classifying driver behaviour …")
    print(f"{sep}")

    predicted_profile, proba = classify_sample(feature_vector)

    print(f"\n  Model : MLP (best performer, 69.15 % accuracy)")

    # Per-class breakdown over the full test set
    y_pred_all = _model.predict(_scaler.transform(X_test))
    report = classification_report(
        y_test, y_pred_all,
        target_names=[LABEL_NAMES[i] for i in range(3)],
    )
    print(f"\n  Per-class performance on held-out test set:")
    for line in report.splitlines():
        print(f"    {line}")

    print(f"\n  Class probabilities for demo sample:")
    for code, name in LABEL_NAMES.items():
        bar = "█" * int(proba[code] * 30)
        print(f"    {name:<14} {proba[code]:.4f}  {bar}")

    print(f"\n  ➜  Predicted profile : {predicted_profile}")
    match = "✓ correct" if predicted_profile == true_profile else f"✗ (true: {true_profile})"
    print(f"     Ground-truth      : {true_profile}  {match}")

    # ------------------------------------------------------------------
    # 3. Route recommendation
    # ------------------------------------------------------------------
    print(f"\n{sep}")
    print("  [3/3]  Computing personalised route …")
    print(f"{sep}")

    origin      = "Home"
    destination = "OfficeE"
    G, all_routes, recommended = recommend_route(predicted_profile, origin, destination)

    print(f"\n  Origin      : {origin}")
    print(f"  Destination : {destination}")
    print(f"\n  Profile description:")
    print(f"    {PROFILE_DESCRIPTIONS[predicted_profile]}")

    print_route(predicted_profile, recommended)

    # Also show how the other profiles would have routed (for comparison)
    print(f"\n  --- All profiles for comparison ---")
    for p in ("Conservative", "Normal", "Spirited"):
        marker = "◄ SELECTED" if p == predicted_profile else ""
        path   = all_routes[p]["path"]
        cost   = all_routes[p]["cost"]
        km     = sum(e[3] for e in all_routes[p]["edges"])
        print(f"  {p:<14} {' → '.join(path)}")
        print(f"               {km:.1f} km  |  weighted cost {cost:.4f}  {marker}")

    # Save the visualisation
    plot_routes(G, all_routes, origin, destination)
    print(f"\n  Route visualisation saved → {ROUTE_FIGURE}")

    print(f"\n{sep}")
    print("  Demo complete.")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
