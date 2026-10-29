
from pathlib import Path

import geopandas as gpd
import pandas as pd

# -------------------------------------------------
# 1. Input and output paths
# -------------------------------------------------

INPUT_FILE = Path(
    "outputs/detections/ais_matching_results.csv"
)

BOUNDARY_DIR = Path("outputs/boundaries")

OUTPUT_FILE = Path(
    "outputs/detections/maritime_risk_results.csv"
)

# -------------------------------------------------
# 2. Load detection results
# -------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required = {
    "detection_id",
    "latitude",
    "longitude",
    "ais_match_status",
}

if not required.issubset(df.columns):
    raise ValueError("Missing required detection columns")

# Convert geographic coordinates into spatial points.
points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["longitude"],
        df["latitude"]
    ),
    crs="EPSG:4326"
)

# -------------------------------------------------
# 3. Load Indian maritime boundary polygons
# -------------------------------------------------

def load_boundary(filename):
    path = BOUNDARY_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing boundary file: {path}"
        )

    boundary = gpd.read_file(path)

    if boundary.crs is None:
        raise ValueError(
            f"Boundary has no coordinate system: {path}"
        )

    return boundary.to_crs("EPSG:4326")


territorial = load_boundary("territorial_sea.geojson")
contiguous = load_boundary("contiguous_zone.geojson")
eez = load_boundary("eez.geojson")

# -------------------------------------------------
# 4. Determine which polygon contains each point
# -------------------------------------------------

def inside_boundary(point, boundary):
    return bool(
        boundary.geometry.intersects(point).any()
    )


def classify_zone(point):
    # Priority is important if boundary polygons overlap.
    if inside_boundary(point, territorial):
        return "Territorial Sea"

    if inside_boundary(point, contiguous):
        return "Contiguous Zone"

    if inside_boundary(point, eez):
        return "Exclusive Economic Zone"

    return "Outside Available Boundaries"


points["maritime_zone"] = points.geometry.apply(
    classify_zone
)

# -------------------------------------------------
# 5. Assign illustrative review priority
# -------------------------------------------------

def review_priority(row):
    # These are synthetic-data demonstration categories.
    # They are NOT assessments of illegal activity.

    if row["ais_match_status"] == "SIMULATED_AIS_MATCH":
        return "Low - Demo AIS Match"

    if row["maritime_zone"] == "Territorial Sea":
        return "High - Manual Review"

    if row["maritime_zone"] == "Contiguous Zone":
        return "Medium - Manual Review"

    return "Unverified - Manual Review"


points["review_priority"] = points.apply(
    review_priority,
    axis=1
)

points["analysis_mode"] = "SYNTHETIC_AIS_DEMONSTRATION"

# -------------------------------------------------
# 6. Save CSV results
# -------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

points.drop(
    columns="geometry"
).to_csv(
    OUTPUT_FILE,
    index=False
)

# -------------------------------------------------
# 7. Print summary
# -------------------------------------------------

print("\nMARITIME BOUNDARY ANALYSIS COMPLETED")
print("-----------------------------------")
print("Total SAR candidates:", len(points))

print("\nCandidates by maritime zone:")
print(points["maritime_zone"].value_counts())

print("\nIllustrative review priorities:")
print(points["review_priority"].value_counts())

print("\nSaved:", OUTPUT_FILE)
print(
    "WARNING: AIS data is synthetic; "
    "review priorities do not indicate illegal activity."
)
