# PCA-Based Maritime Surveillance System

## Project Overview
This project analyzes Sentinel-1 Synthetic Aperture Radar (SAR)
imagery to identify preliminary vessel candidates in the
Mumbai offshore region of the Arabian Sea.

The system uses Principal Component Analysis (PCA),
radar backscatter analysis, geographic boundary classification,
and AIS matching demonstrations.

An interactive Plotly Dash dashboard visualizes the results.

## Technologies Used
- Python
- NumPy and Pandas
- Rasterio
- OpenCV
- Scikit-learn PCA
- GeoPandas and Shapely
- Plotly and Dash
- ESA SNAP for SAR preprocessing

## Methodology
1. Download Sentinel-1 SAR imagery.
2. Perform SAR preprocessing using ESA SNAP.
3. Extract radar backscatter and local statistical features.
4. Apply PCA for dimensionality reduction.
5. Generate preliminary SAR vessel candidates.
6. Apply candidate filtering.
7. Demonstrate SAR-AIS matching using synthetic AIS records.
8. Classify candidates by maritime boundary zone.
9. Display results using an interactive Dash dashboard.

## Project Results
- SAR candidates after filtering: 72
- PCA cumulative explained variance (3 components): approximately 99.5%
- Synthetic AIS records: 25
- Matches with synthetic AIS: 18
- Candidates without synthetic AIS matches: 54
- Contiguous Zone candidates: 46
- EEZ candidates: 26

## Important Limitations
The 72 SAR detections are preliminary candidates, not
independently confirmed vessels.

AIS records in synthetic_ais_demo.csv are simulated.
The AIS matching results demonstrate the algorithm and
do not validate real vessel identities.

Unmatched candidates do not imply illegal or suspicious activity.
The review-priority categories are illustrative.

## Running the Dashboard
Install the Python dependencies:

    python -m pip install -r requirements.txt

Start the dashboard:

    python app.py

Open:

    http://127.0.0.1:8051/

## Main Output Files
- outputs/pca/PC1.tif
- outputs/pca/PC2.tif
- outputs/pca/PC3.tif
- outputs/pca/explained_variance.csv
- outputs/pca/pca_loadings.csv
- outputs/detections/filtered_vessel_candidates.csv
- outputs/detections/ais_matching_results.csv
- outputs/detections/maritime_risk_results.csv

## Study Area
Mumbai offshore, Arabian Sea, India.

Satellite acquisition date: 30 September 2026.
