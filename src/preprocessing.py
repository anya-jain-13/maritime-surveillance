# Import Path to work with file and folder locations.
from pathlib import Path

# Import Rasterio to read satellite raster images.
import rasterio

# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Find the folder containing our Sentinel-1 satellite data.
SATELLITE_FOLDER = PROJECT_ROOT / "data" / "satellite_images"

# Find the Sentinel-1 SAFE product folder.
safe_folders = list(SATELLITE_FOLDER.glob("*.SAFE"))

# Stop if no satellite product is found.
if not safe_folders:
    raise FileNotFoundError("No Sentinel-1 SAFE folder found!")

# Select our downloaded satellite product.
safe_folder = safe_folders[0]

# Locate the radar measurement folder.
measurement_folder = safe_folder / "measurement"

# Find the VV and VH radar images.
vv_files = list(measurement_folder.glob("*-vv-*.tiff"))
vh_files = list(measurement_folder.glob("*-vh-*.tiff"))

# Confirm that both radar images exist.
if not vv_files or not vh_files:
    raise FileNotFoundError("VV or VH radar image is missing!")

# Open the VV radar image.
with rasterio.open(vv_files[0]) as src:

    # Display basic satellite image information.
    print("Satellite product:", safe_folder.name)
    print("Image width:", src.width)
    print("Image height:", src.height)
    print("Number of bands:", src.count)
    print("Pixel data type:", src.dtypes)
    print("Coordinate system:", src.crs)
    print("Image bounds:", src.bounds)

# Confirm that the VH image can also be opened.
with rasterio.open(vh_files[0]) as src:
    print("VH image dimensions:", src.width, "x", src.height)
    print("VH pixel data type:", src.dtypes)