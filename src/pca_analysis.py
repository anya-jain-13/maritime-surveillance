
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------
# 1. Define project paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "subset_Mumbai_SAR_TerrainCorrected.tif"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pca"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# 2. Read the satellite image
# -------------------------------------------------------

def load_sar_image():
    """Load the VH and VV bands from the GeoTIFF."""

    with rasterio.open(INPUT_FILE) as src:
        # SNAP exported VH first and VV second.
        vh = src.read(1).astype(np.float32)
        vv = src.read(2).astype(np.float32)

        # Preserve spatial metadata for output rasters.
        profile = src.profile.copy()

        # Create a mask of valid satellite pixels.
        valid = (
            np.isfinite(vh)
            & np.isfinite(vv)
            & (vh > 0)
            & (vv > 0)
        )

        if src.nodata is not None:
            valid &= (vh != src.nodata) & (vv != src.nodata)

    print("Satellite image loaded successfully.")
    print("Image dimensions:", vh.shape)
    print("Valid pixels:", int(valid.sum()))

    return vh, vv, valid, profile


# -------------------------------------------------------
# 3. Convert radar intensity into decibels
# -------------------------------------------------------

def convert_to_db(image):
    """Convert linear radar backscatter to dB."""

    return 10.0 * np.log10(np.maximum(image, 1e-10))


# -------------------------------------------------------
# 4. Extract image features
# -------------------------------------------------------

def extract_features(vh, vv, valid):
    """Build a feature matrix for PCA."""

    vh_db = convert_to_db(vh)
    vv_db = convert_to_db(vv)

    # Use neutral values outside valid image coverage
    # so they do not distort local texture calculations.
    vh_work = np.where(valid, vh_db, 0).astype(np.float32)
    vv_work = np.where(valid, vv_db, 0).astype(np.float32)

    # Smooth VV brightness over a 5x5 neighbourhood.
    local_mean = cv2.blur(vv_work, (5, 5))

    # Compute local variance as a texture feature.
    local_square_mean = cv2.blur(vv_work ** 2, (5, 5))
    local_variance = np.maximum(
        local_square_mean - local_mean ** 2,
        0
    )

    # Polarization difference can help distinguish
    # different radar scattering characteristics.
    polarization_difference = vv_work - vh_work

    features = np.column_stack([
        vh_db[valid],
        vv_db[valid],
        local_mean[valid],
        local_variance[valid],
        polarization_difference[valid],
    ]).astype(np.float32)

    feature_names = [
        "VH_dB",
        "VV_dB",
        "Local_Mean_VV",
        "Local_Variance_VV",
        "VV_minus_VH_dB",
    ]

    return features, feature_names


# -------------------------------------------------------
# 5. Apply Principal Component Analysis
# -------------------------------------------------------

def run_pca(features):
    """Standardize features and compute three PCs."""

    scaler = StandardScaler()

    # Fit the scaler using a reproducible sample.
    rng = np.random.default_rng(42)
    sample_size = min(200_000, len(features))
    sample_indices = rng.choice(
        len(features),
        size=sample_size,
        replace=False
    )

    scaler.fit(features[sample_indices])

    # Transform the sample to learn PCA directions.
    sample_scaled = scaler.transform(
        features[sample_indices]
    )

    pca = PCA(n_components=3, random_state=42)
    pca.fit(sample_scaled)

    # Transform the complete image in batches to
    # reduce peak memory usage.
    components = np.empty(
        (len(features), 3),
        dtype=np.float32
    )

    batch_size = 250_000

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))

        scaled_batch = scaler.transform(
            features[start:end]
        )

        components[start:end] = pca.transform(
            scaled_batch
        ).astype(np.float32)

    return components, pca


# -------------------------------------------------------
# 6. Save PCA components as georeferenced images
# -------------------------------------------------------

def save_components(components, valid, profile):
    """Save PC1, PC2 and PC3 as separate GeoTIFFs."""

    height, width = valid.shape

    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=-9999.0,
        compress="deflate"
    )

    for index in range(3):
        image = np.full(
            (height, width),
            -9999.0,
            dtype=np.float32
        )

        image[valid] = components[:, index]

        output_path = OUTPUT_DIR / f"PC{index + 1}.tif"

        with rasterio.open(
            output_path,
            "w",
            **output_profile
        ) as dst:
            dst.write(image, 1)

        print("Saved:", output_path.name)


# -------------------------------------------------------
# 7. Save explained variance and PCA loadings
# -------------------------------------------------------

def save_pca_statistics(pca, feature_names):
    """Export PCA results for analysis and reporting."""

    variance_df = pd.DataFrame({
        "Component": ["PC1", "PC2", "PC3"],
        "Explained_Variance_Ratio":
            pca.explained_variance_ratio_,
        "Cumulative_Variance":
            np.cumsum(pca.explained_variance_ratio_),
    })

    variance_df.to_csv(
        OUTPUT_DIR / "explained_variance.csv",
        index=False
    )

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=["PC1", "PC2", "PC3"]
    )

    loadings_df.to_csv(
        OUTPUT_DIR / "pca_loadings.csv"
    )

    print("\nExplained variance:")
    print(variance_df.to_string(index=False))


# -------------------------------------------------------
# 8. Main execution
# -------------------------------------------------------

def main():
    print("Starting maritime satellite PCA...\n")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Satellite image not found: {INPUT_FILE}"
        )

    vh, vv, valid, profile = load_sar_image()

    if valid.sum() < 100:
        raise ValueError(
            "Too few valid satellite pixels for PCA."
        )

    print("\nExtracting satellite image features...")
    features, feature_names = extract_features(
        vh, vv, valid
    )

    print("Feature matrix shape:", features.shape)

    print("\nRunning PCA...")
    components, pca = run_pca(features)

    print("\nSaving PCA images...")
    save_components(components, valid, profile)

    save_pca_statistics(pca, feature_names)

    print("\nPCA processing completed successfully!")


if __name__ == "__main__":
    main()
