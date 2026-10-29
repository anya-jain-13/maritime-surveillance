
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy


# -------------------------------------------------------
# 1. Define project paths
# -------------------------------------------------------

# Locate the root folder of the maritime project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Terrain-corrected Sentinel-1 image.
SAR_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "subset_Mumbai_SAR_TerrainCorrected.tif"
)

# PCA component produced in the previous stage.
PCA_FILE = PROJECT_ROOT / "outputs" / "pca" / "PC3.tif"

# Folder for vessel detection results.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "detections"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# 2. Detection parameters
# -------------------------------------------------------

# Estimate local background over a 101x101 window.
BACKGROUND_WINDOW = 101

# A candidate must stand out from local sea clutter.
MIN_LOCAL_CONTRAST_DB = 5.0

# Require at least this many pixels in a candidate.
MIN_AREA_PIXELS = 3

# Reject very large bright regions.
MAX_AREA_PIXELS = 350

# Limit excessively elongated detections.
MAX_ASPECT_RATIO = 8.0

# Small morphological opening removes isolated noise.
MORPH_KERNEL_SIZE = 2


# -------------------------------------------------------
# 3. Load satellite data
# -------------------------------------------------------

def load_satellite_data():
    """Load VV, VH, PCA and geographic metadata."""

    with rasterio.open(SAR_FILE) as src:

        # SNAP exported VH as band 1 and VV as band 2.
        vh = src.read(1).astype(np.float32)
        vv = src.read(2).astype(np.float32)

        # Store the geographic transform and CRS.
        transform = src.transform
        crs = src.crs

        # Identify valid radar measurements.
        valid = (
            np.isfinite(vh)
            & np.isfinite(vv)
            & (vh > 0)
            & (vv > 0)
        )

        if src.nodata is not None:
            valid &= (
                (vh != src.nodata)
                & (vv != src.nodata)
            )

    # Load the third principal component.
    with rasterio.open(PCA_FILE) as src:

        pc3 = src.read(1).astype(np.float32)

        if pc3.shape != vv.shape:
            raise ValueError(
                "PCA and SAR image dimensions do not match."
            )

        if src.crs != crs or not src.transform.almost_equals(
            transform
        ):
            raise ValueError(
                "PCA and SAR image georeferencing does not match."
            )

        pc3_nodata = src.nodata

        pc3_valid = np.isfinite(pc3)

        if pc3_nodata is not None:
            pc3_valid &= pc3 != pc3_nodata

        valid &= pc3_valid

    print("Satellite image loaded.")
    print("Dimensions:", vv.shape)
    print("Valid pixels:", int(valid.sum()))
    print("Coordinate system:", crs)

    return vv, vh, pc3, valid, transform


# -------------------------------------------------------
# 4. Convert radar intensity into decibels
# -------------------------------------------------------

def to_db(image):
    """Convert linear SAR backscatter to decibels."""

    return 10.0 * np.log10(
        np.maximum(image, 1e-10)
    )


# -------------------------------------------------------
# 5. Detect bright objects using local contrast
# -------------------------------------------------------

def create_candidate_mask(vv, valid):
    """Find locally bright radar targets."""

    vv_db = to_db(vv)

    # Exclude invalid pixels from background estimates.
    valid_float = valid.astype(np.float32)

    image = np.where(
        valid,
        vv_db,
        0
    ).astype(np.float32)

    # Compute a valid-pixel-weighted local background.
    local_sum = cv2.boxFilter(
        image,
        ddepth=-1,
        ksize=(BACKGROUND_WINDOW, BACKGROUND_WINDOW),
        normalize=False,
        borderType=cv2.BORDER_REFLECT
    )

    local_count = cv2.boxFilter(
        valid_float,
        ddepth=-1,
        ksize=(BACKGROUND_WINDOW, BACKGROUND_WINDOW),
        normalize=False,
        borderType=cv2.BORDER_REFLECT
    )

    local_mean = local_sum / np.maximum(
        local_count,
        1.0
    )

    # Bright targets should exceed nearby background.
    contrast = vv_db - local_mean

    # Avoid detections at boundaries with little valid data.
    enough_background = (
        local_count
        >= BACKGROUND_WINDOW * BACKGROUND_WINDOW * 0.8
    )

    candidate_mask = (
        (contrast >= MIN_LOCAL_CONTRAST_DB)
        & valid
        & enough_background
    )

    # Convert Boolean mask into OpenCV's uint8 format.
    mask_uint8 = (
        candidate_mask.astype(np.uint8) * 255
    )

    # Remove isolated pixels using morphological opening.
    kernel = np.ones(
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        dtype=np.uint8
    )

    cleaned_mask = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_OPEN,
        kernel
    )

    return cleaned_mask, contrast, vv_db


# -------------------------------------------------------
# 6. Extract vessel candidates
# -------------------------------------------------------

def extract_candidates(
    mask,
    contrast,
    vv_db,
    pc3,
    transform
):
    """Extract connected objects and geographic positions."""

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    detections = []

    # Label zero represents the background.
    for label in range(1, num_labels):

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])

        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        area = int(stats[label, cv2.CC_STAT_AREA])

        # Reject candidates outside the desired size range.
        if not MIN_AREA_PIXELS <= area <= MAX_AREA_PIXELS:
            continue

        aspect_ratio = max(
            width / max(height, 1),
            height / max(width, 1)
        )

        if aspect_ratio > MAX_ASPECT_RATIO:
            continue

        # Obtain pixels belonging to this connected object.
        component = (
            labels[y:y + height, x:x + width] == label
        )

        contrast_values = contrast[
            y:y + height,
            x:x + width
        ][component]

        vv_values = vv_db[
            y:y + height,
            x:x + width
        ][component]

        pc3_values = pc3[
            y:y + height,
            x:x + width
        ][component]

        # Calculate the center of the candidate.
        center_x = float(centroids[label][0])
        center_y = float(centroids[label][1])

        # Convert pixel coordinates to geographic coordinates.
        longitude, latitude = xy(
            transform,
            center_y,
            center_x
        )

        detections.append({
            "detection_id": len(detections) + 1,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "pixel_x": center_x,
            "pixel_y": center_y,
            "area_pixels": area,
            "width_pixels": width,
            "height_pixels": height,
            "aspect_ratio": round(aspect_ratio, 3),
            "mean_contrast_db": round(
                float(np.mean(contrast_values)),
                3
            ),
            "max_contrast_db": round(
                float(np.max(contrast_values)),
                3
            ),
            "mean_vv_db": round(
                float(np.mean(vv_values)),
                3
            ),
            "mean_pc3": round(
                float(np.mean(pc3_values)),
                3
            ),
            "status": "Unverified SAR candidate"
        })

    return pd.DataFrame(detections)


# -------------------------------------------------------
# 7. Save detections and preview
# -------------------------------------------------------

def save_results(
    detections,
    mask,
    vv_db,
    valid
):
    """Save candidate coordinates and visualization."""

    csv_path = OUTPUT_DIR / "vessel_candidates.csv"

    detections.to_csv(
        csv_path,
        index=False
    )

    print("Saved detections:", csv_path)

    # Scale valid VV pixels to grayscale for display.
    lower, upper = np.percentile(
        vv_db[valid],
        [2, 98]
    )

    scaled = np.clip(
        (vv_db - lower) / max(upper - lower, 1e-6),
        0,
        1
    )

    grayscale = (
        scaled * 255
    ).astype(np.uint8)

    grayscale[~valid] = 0

    # Convert grayscale to BGR for colored annotations.
    preview = cv2.cvtColor(
        grayscale,
        cv2.COLOR_GRAY2BGR
    )

    # Draw red circles around vessel candidates.
    for _, detection in detections.iterrows():

        center = (
            int(round(detection["pixel_x"])),
            int(round(detection["pixel_y"]))
        )

        cv2.circle(
            preview,
            center,
            7,
            (0, 0, 255),
            1
        )

    preview_path = OUTPUT_DIR / "vessel_detection_preview.png"

    cv2.imwrite(
        str(preview_path),
        preview
    )

    mask_path = OUTPUT_DIR / "candidate_mask.png"

    cv2.imwrite(
        str(mask_path),
        mask
    )

    print("Saved preview:", preview_path)
    print("Saved mask:", mask_path)


# -------------------------------------------------------
# 8. Main program
# -------------------------------------------------------

def main():
    print("Starting SAR vessel candidate detection...\n")

    if not SAR_FILE.exists():
        raise FileNotFoundError(
            f"SAR image not found: {SAR_FILE}"
        )

    if not PCA_FILE.exists():
        raise FileNotFoundError(
            f"PCA image not found: {PCA_FILE}"
        )

    vv, vh, pc3, valid, transform = (
        load_satellite_data()
    )

    print("\nCalculating local radar contrast...")

    mask, contrast, vv_db = create_candidate_mask(
        vv,
        valid
    )

    print("\nExtracting candidate objects...")

    detections = extract_candidates(
        mask,
        contrast,
        vv_db,
        pc3,
        transform
    )

    print("Number of candidates:", len(detections))

    save_results(
        detections,
        mask,
        vv_db,
        valid
    )

    print("\nVessel candidate detection completed!")

    if not detections.empty:
        print("\nFirst 10 candidates:")
        print(detections.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
