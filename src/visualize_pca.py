
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Find the project's root folder.
ROOT = Path(__file__).resolve().parent.parent

# Location of the saved PCA GeoTIFFs.
PCA_DIR = ROOT / "outputs" / "pca"

# Create a figure with three panels.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Display the first three principal components.
for i, ax in enumerate(axes, start=1):

    # Open the PCA GeoTIFF.
    with rasterio.open(PCA_DIR / f"PC{i}.tif") as src:
        image = src.read(1)
        nodata = src.nodata

    # Identify valid pixels.
    valid = np.isfinite(image)

    if nodata is not None:
        valid &= image != nodata

    # Set invalid pixels to NaN for display.
    display = np.where(valid, image, np.nan)

    # Use percentile-based contrast for visualization.
    low, high = np.percentile(
        image[valid], [2, 98]
    )

    ax.imshow(
        display,
        cmap="gray",
        vmin=low,
        vmax=high
    )

    ax.set_title(f"Principal Component {i}")
    ax.axis("off")

# Save the comparison image.
plt.tight_layout()

output = PCA_DIR / "pca_comparison.png"
plt.savefig(output, dpi=180, bbox_inches="tight")

print("PCA comparison saved:", output)

# Display the visualization.
plt.show()
