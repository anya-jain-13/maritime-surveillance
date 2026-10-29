
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. File paths and configuration
# --------------------------------------------------

SAR_FILE = Path("outputs/detections/filtered_vessel_candidates.csv")
AIS_FILE = Path("data/ais_data/synthetic_ais_demo.csv")
OUTPUT_FILE = Path("outputs/detections/ais_matching_results.csv")

SATELLITE_TIME = pd.Timestamp("2026-09-30T01:02:00Z")

MAX_DISTANCE_KM = 1.0
MAX_TIME_DIFFERENCE_MIN = 10


# --------------------------------------------------
# 2. Haversine distance
# Calculates distance between two GPS coordinates
# --------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371.0088

    lat1, lon1, lat2, lon2 = np.radians(
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * radius_km * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


# --------------------------------------------------
# 3. Load satellite and AIS datasets
# --------------------------------------------------

sar = pd.read_csv(SAR_FILE)
ais = pd.read_csv(AIS_FILE)

required_sar = {"detection_id", "latitude", "longitude"}
required_ais = {"mmsi", "timestamp", "latitude", "longitude"}

if not required_sar.issubset(sar.columns):
    raise ValueError("SAR dataset is missing required columns")

if not required_ais.issubset(ais.columns):
    raise ValueError("AIS dataset is missing required columns")

# Preserve identifiers as strings.
ais["mmsi"] = ais["mmsi"].astype(str)

# Parse timestamps as UTC.
ais["timestamp"] = pd.to_datetime(
    ais["timestamp"], utc=True, errors="coerce"
)

# Calculate time difference from satellite acquisition.
ais["time_difference_min"] = (
    (ais["timestamp"] - SATELLITE_TIME)
    .abs()
    .dt.total_seconds() / 60
)

# Retain only AIS positions within the allowed time window.
valid_ais = ais[
    ais["time_difference_min"] <= MAX_TIME_DIFFERENCE_MIN
].copy()


# --------------------------------------------------
# 4. Find potential geographic matches
# --------------------------------------------------

potential_matches = []

for _, detection in sar.iterrows():

    for _, vessel in valid_ais.iterrows():

        distance = haversine(
            detection["latitude"],
            detection["longitude"],
            vessel["latitude"],
            vessel["longitude"]
        )

        if distance <= MAX_DISTANCE_KM:
            potential_matches.append({
                "detection_id": detection["detection_id"],
                "mmsi": vessel["mmsi"],
                "distance_km": float(distance),
                "time_difference_min": float(
                    vessel["time_difference_min"]
                )
            })


# --------------------------------------------------
# 5. Assign closest one-to-one matches
# --------------------------------------------------

potential_matches.sort(
    key=lambda item: (
        item["distance_km"],
        item["time_difference_min"]
    )
)

assigned_detections = set()
assigned_mmsi = set()
confirmed_pairs = {}

for match in potential_matches:

    detection_id = match["detection_id"]
    mmsi = match["mmsi"]

    if detection_id in assigned_detections:
        continue

    if mmsi in assigned_mmsi:
        continue

    confirmed_pairs[detection_id] = match

    assigned_detections.add(detection_id)
    assigned_mmsi.add(mmsi)


# --------------------------------------------------
# 6. Build results for every SAR candidate
# --------------------------------------------------

results = []

for _, detection in sar.iterrows():

    detection_id = detection["detection_id"]
    match = confirmed_pairs.get(detection_id)

    result = detection.to_dict()

    if match is not None:
        result["ais_match_status"] = "SIMULATED_AIS_MATCH"
        result["matched_mmsi"] = match["mmsi"]
        result["match_distance_km"] = round(
            match["distance_km"], 4
        )
        result["time_difference_min"] = round(
            match["time_difference_min"], 2
        )

    else:
        result["ais_match_status"] = "NO_SIMULATED_AIS_MATCH"
        result["matched_mmsi"] = ""
        result["match_distance_km"] = np.nan
        result["time_difference_min"] = np.nan

    result["ais_data_source"] = "SYNTHETIC_DEMO"

    results.append(result)


# --------------------------------------------------
# 7. Save output
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

matched_count = (
    results_df["ais_match_status"]
    == "SIMULATED_AIS_MATCH"
).sum()

print("\nSAR-AIS MATCHING COMPLETED")
print("--------------------------")
print("SAR candidates:", len(sar))
print("Synthetic AIS records:", len(ais))
print("Simulated AIS matches:", matched_count)
print("Unmatched SAR candidates:", len(sar) - matched_count)
print("Maximum distance:", MAX_DISTANCE_KM, "km")
print("Maximum time difference:", MAX_TIME_DIFFERENCE_MIN, "minutes")
print("Saved:", OUTPUT_FILE)
print("WARNING: Matches use simulated AIS, not real vessel observations.")
