"""
preprocess.py — Data Loading & Feature Extraction

Walks UAH-DRIVESET-v1, reads the raw accelerometer, GPS, and lane detection
files for each trip, and builds overlapping 5-second sliding windows. Each
window becomes one row in processed_dataset.csv with 15 extracted features.

The UAH dataset uses DROWSY and AGGRESSIVE labels — I remapped them to better
reflect driving style for a routing use case:
  NORMAL     → Normal       (0)
  DROWSY     → Conservative (1)
  AGGRESSIVE → Spirited     (2)
"""

import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "datasets", "UAH-DRIVESET-v1")
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, "processed_dataset.csv")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "NORMAL":     ("Normal",       0),
    "DROWSY":     ("Conservative", 1),
    "AGGRESSIVE": ("Spirited",     2),
}

WINDOW_SIZE = 50   # rows per window  (~5 s at 10 Hz)
WINDOW_STEP = 25   # rows between window starts (50 % overlap)
BRAKING_THRESHOLD = -0.3   # accel_x_filtered threshold for a braking event


# ---------------------------------------------------------------------------
# Folder-name parsing
# ---------------------------------------------------------------------------
def parse_trip_folder(folder_name: str):
    """
    Extract (label_str, road_type_int) from a trip folder name.

    Expected format:  <timestamp>-<dist>km-<driver>-<LABEL>[<N>]-<ROADTYPE>
    e.g.  20151110175712-16km-D1-NORMAL1-SECONDARY
          20151111125233-24km-D1-AGGRESSIVE-MOTORWAY

    Returns (label_str, road_type) or (None, None) if pattern doesn't match.
    """
    # The label token may have a trailing digit (NORMAL1, NORMAL2, etc.)
    pattern = r"-(?P<label>NORMAL\d?|AGGRESSIVE|DROWSY)-(?P<road>MOTORWAY|SECONDARY)"
    m = re.search(pattern, folder_name)
    if not m:
        return None, None

    raw_label = re.sub(r"\d+$", "", m.group("label"))   # strip trailing digit
    road_type = 1 if m.group("road") == "MOTORWAY" else 0
    return raw_label, road_type


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------
def load_accelerometers(path: str) -> pd.DataFrame | None:
    """Load RAW_ACCELEROMETERS.txt and keep only rows where the system was active."""
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1, 5, 6, 7],
                         names=["timestamp", "system_activated",
                                "accel_x", "accel_y", "accel_z"])
        df = df[df["system_activated"] == 1].drop(columns=["system_activated"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"    [WARN] Could not load accelerometers from {path}: {e}")
        return None


def load_gps(path: str) -> pd.DataFrame | None:
    """Load RAW_GPS.txt and return timestamp + speed_kmh."""
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1],
                         names=["timestamp", "speed_kmh"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"    [WARN] Could not load GPS from {path}: {e}")
        return None


def load_lane_detection(path: str) -> pd.DataFrame | None:
    """Load PROC_LANE_DETECTION.txt. Rows with -9 are dropped (no lane detected)."""
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1],
                         names=["timestamp", "lane_deviation"])
        # -9 is the no-detection sentinel
        df = df[df["lane_deviation"] != -9].copy()
        df["lane_deviation"] = df["lane_deviation"].abs()
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"    [WARN] Could not load lane detection from {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Merge sensor streams
# ---------------------------------------------------------------------------
def merge_sensors(accel_df: pd.DataFrame, gps_df: pd.DataFrame,
                  lane_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge sensor streams onto a shared timeline using merge_asof.
    Accelerometers run at ~10 Hz, GPS at ~1 Hz, so each accel row gets matched
    to the nearest GPS reading within 0.5 s. Lane detection is handled the same
    way; rows with no lane data default to lane_deviation = 0.
    """
    merged = pd.merge_asof(
        accel_df,
        gps_df,
        on="timestamp",
        direction="nearest",
        tolerance=0.5,
    )

    if lane_df is not None and not lane_df.empty:
        merged = pd.merge_asof(
            merged,
            lane_df,
            on="timestamp",
            direction="nearest",
            tolerance=0.5,
        )
    else:
        merged["lane_deviation"] = 0.0

    # Drop rows where GPS match failed (speed_kmh is NaN after merge)
    merged = merged.dropna(subset=["speed_kmh"])
    merged["lane_deviation"] = merged["lane_deviation"].fillna(0.0)

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature extraction from a single window
# ---------------------------------------------------------------------------
def extract_features(window: pd.DataFrame, road_type: int) -> dict:
    """
    Compute 15 scalar features from a single time-window DataFrame.
    """
    ax = window["accel_x"].values
    ay = window["accel_y"].values
    az = window["accel_z"].values
    spd = window["speed_kmh"].values
    lane = window["lane_deviation"].values

    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

    # Fraction of rows in the window that qualify as a braking event
    braking_event_rate = np.mean(ax < BRAKING_THRESHOLD)

    return {
        "speed_mean":            float(np.mean(spd)),
        "speed_std":             float(np.std(spd)),
        "speed_max":             float(np.max(spd)),
        "speed_range":           float(np.max(spd) - np.min(spd)),
        "accel_x_mean":          float(np.mean(ax)),
        "accel_x_std":           float(np.std(ax)),
        "accel_x_min":           float(np.min(ax)),
        "accel_y_mean":          float(np.mean(ay)),
        "accel_y_std":           float(np.std(ay)),
        "accel_y_max":           float(np.max(np.abs(ay))),
        "accel_magnitude_mean":  float(np.mean(accel_mag)),
        "accel_magnitude_std":   float(np.std(accel_mag)),
        "braking_event_rate":    float(braking_event_rate),
        "lane_deviation_mean":   float(np.mean(lane)),
        "road_type":             int(road_type),
    }


# ---------------------------------------------------------------------------
# Sliding-window builder for one trip
# ---------------------------------------------------------------------------
def build_windows(merged: pd.DataFrame, road_type: int,
                  label_encoded: int, driver: str) -> list[dict]:
    """Slide a fixed-size window over the trip data and extract features from each window."""
    rows = []
    n = len(merged)

    for start in range(0, n - WINDOW_SIZE + 1, WINDOW_STEP):
        window = merged.iloc[start: start + WINDOW_SIZE]
        features = extract_features(window, road_type)
        features["label"] = label_encoded
        features["driver"] = driver
        rows.append(features)

    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    all_windows: list[dict] = []
    trips_loaded = 0
    trips_skipped = 0

    print(f"Scanning dataset at: {DATASET_ROOT}\n")

    # Walk driver directories (D1–D6), then trip subdirectories
    for driver in sorted(os.listdir(DATASET_ROOT)):
        driver_path = os.path.join(DATASET_ROOT, driver)
        if not os.path.isdir(driver_path):
            continue

        for trip_name in sorted(os.listdir(driver_path)):
            trip_path = os.path.join(driver_path, trip_name)
            if not os.path.isdir(trip_path):
                continue

            # --- Parse label and road type from folder name ----------------
            raw_label, road_type = parse_trip_folder(trip_name)
            if raw_label is None or raw_label not in LABEL_MAP:
                print(f"  [SKIP] Unrecognised label in: {trip_name}")
                trips_skipped += 1
                continue

            label_name, label_encoded = LABEL_MAP[raw_label]

            # --- Check required files --------------------------------------
            accel_file = os.path.join(trip_path, "RAW_ACCELEROMETERS.txt")
            gps_file   = os.path.join(trip_path, "RAW_GPS.txt")
            lane_file  = os.path.join(trip_path, "PROC_LANE_DETECTION.txt")

            if not os.path.isfile(accel_file) or not os.path.isfile(gps_file):
                print(f"  [SKIP] Missing required files in: {trip_name}")
                trips_skipped += 1
                continue

            # --- Load files ------------------------------------------------
            print(f"  Loading: {driver}/{trip_name}")
            accel_df = load_accelerometers(accel_file)
            gps_df   = load_gps(gps_file)

            if accel_df is None or accel_df.empty or gps_df is None or gps_df.empty:
                print(f"    [SKIP] Empty or unreadable sensor data.")
                trips_skipped += 1
                continue

            lane_df = None
            if os.path.isfile(lane_file):
                lane_df = load_lane_detection(lane_file)

            # --- Merge & window --------------------------------------------
            merged = merge_sensors(accel_df, gps_df, lane_df)

            if len(merged) < WINDOW_SIZE:
                print(f"    [SKIP] Too few merged rows ({len(merged)}) for a window.")
                trips_skipped += 1
                continue

            windows = build_windows(merged, road_type, label_encoded, driver)
            all_windows.extend(windows)
            trips_loaded += 1
            print(f"    -> {len(windows)} windows  |  label={label_name}  |  "
                  f"road={'MOTORWAY' if road_type else 'SECONDARY'}")

    # --- Assemble DataFrame ------------------------------------------------
    if not all_windows:
        print("\n[ERROR] No windows extracted — check dataset path and file formats.")
        return

    df = pd.DataFrame(all_windows)

    # Reorder columns: features first, label last
    feature_cols = [
        "speed_mean", "speed_std", "speed_max", "speed_range",
        "accel_x_mean", "accel_x_std", "accel_x_min",
        "accel_y_mean", "accel_y_std", "accel_y_max",
        "accel_magnitude_mean", "accel_magnitude_std",
        "braking_event_rate",
        "lane_deviation_mean",
        "road_type",
        "label",
        "driver",
    ]
    df = df[feature_cols]

    # --- Save --------------------------------------------------------------
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved processed dataset to: {OUTPUT_PATH}")

    # --- Summary statistics ------------------------------------------------
    label_names = {0: "Normal", 1: "Conservative", 2: "Spirited"}
    total = len(df)

    print(f"\n{'='*55}")
    print(f"  Trips loaded : {trips_loaded}")
    print(f"  Trips skipped: {trips_skipped}")
    print(f"  Total windows: {total}")
    print(f"\n  Class distribution:")
    print(f"  {'Label':<16} {'Encoded':>7}  {'Count':>7}  {'Pct':>7}")
    print(f"  {'-'*45}")
    for code, name in label_names.items():
        count = int((df["label"] == code).sum())
        pct   = 100.0 * count / total if total > 0 else 0.0
        print(f"  {name:<16} {code:>7}  {count:>7}  {pct:>6.1f}%")
    print(f"{'='*55}")

    print(f"\n  Sample (5 rows):")
    print(df.sample(5, random_state=42).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
