
# ============================================================
# MARITIME SURVEILLANCE DASHBOARD
# PCA-Based Sentinel-1 SAR Vessel Candidate Detection
# ============================================================

from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
from shapely.geometry import box


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

BOUNDARY_DIR = ROOT / "outputs" / "boundaries"
DETECTION_DIR = ROOT / "outputs" / "detections"
PCA_DIR = ROOT / "outputs" / "pca"

RISK_FILE = DETECTION_DIR / "maritime_risk_results.csv"
VARIANCE_FILE = PCA_DIR / "explained_variance.csv"
LOADINGS_FILE = PCA_DIR / "pca_loadings.csv"


# ============================================================
# 2. LOAD SAR DETECTION AND AIS RESULTS
# ============================================================

detections = pd.read_csv(RISK_FILE)

required_columns = {
    "detection_id",
    "latitude",
    "longitude",
    "maritime_zone",
    "ais_match_status",
    "review_priority",
}

missing = required_columns - set(detections.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

detections["latitude"] = pd.to_numeric(
    detections["latitude"],
    errors="coerce"
)

detections["longitude"] = pd.to_numeric(
    detections["longitude"],
    errors="coerce"
)

detections = detections.dropna(
    subset=["latitude", "longitude"]
).copy()

for column in ["max_contrast_db", "area_pixels"]:
    if column not in detections.columns:
        detections[column] = float("nan")


# ============================================================
# 3. MARITIME BOUNDARIES
# ============================================================

ZONES = {
    "territorial_sea": {
        "name": "Territorial Sea (12 NM)",
        "color": "#ef4444"
    },
    "contiguous_zone": {
        "name": "Contiguous Zone (24 NM)",
        "color": "#f59e0b"
    },
    "eez": {
        "name": "Exclusive Economic Zone",
        "color": "#38bdf8"
    }
}

BOUNDARIES = {}

for zone_id in ZONES:

    file_path = BOUNDARY_DIR / f"{zone_id}.geojson"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing boundary: {file_path}"
        )

    BOUNDARIES[zone_id] = (
        gpd.read_file(file_path).to_crs(epsg=4326)
    )


# ============================================================
# 4. SATELLITE STUDY AREA
# ============================================================

# Actual bounds of the processed Sentinel-1 image.

WEST = 72.17371910726854
SOUTH = 18.800105316807873
EAST = 72.57634401761091
NORTH = 19.00007029905288

study_area = box(WEST, SOUTH, EAST, NORTH)


# ============================================================
# 5. CREATE INTERACTIVE MAP
# ============================================================

def create_map(filtered_df):

    fig = go.Figure()

    # Draw maritime boundaries.
    for zone_id, settings in ZONES.items():

        gdf = BOUNDARIES[zone_id]

        first_trace = True

        for geometry in gdf.geometry:

            if geometry is None or geometry.is_empty:
                continue

            if geometry.geom_type == "MultiPolygon":
                polygons = list(geometry.geoms)

            elif geometry.geom_type == "Polygon":
                polygons = [geometry]

            else:
                continue

            for polygon in polygons:

                lon, lat = polygon.exterior.xy

                fig.add_trace(
                    go.Scattermap(
                        lon=list(lon),
                        lat=list(lat),
                        mode="lines",
                        line={
                            "width": 2,
                            "color": settings["color"]
                        },
                        name=settings["name"],
                        legendgroup=zone_id,
                        showlegend=first_trace,
                        hoverinfo="skip"
                    )
                )

                first_trace = False

    # Draw the processed satellite study area.
    study_lon, study_lat = study_area.exterior.xy

    fig.add_trace(
        go.Scattermap(
            lon=list(study_lon),
            lat=list(study_lat),
            mode="lines",
            line={
                "width": 3,
                "color": "#22c55e"
            },
            name="Processed SAR Study Area",
            hoverinfo="skip"
        )
    )

    # SAR candidate marker categories.
    marker_groups = [
        {
            "status": "SIMULATED_AIS_MATCH",
            "label": "Synthetic AIS Match",
            "color": "#22c55e"
        },
        {
            "status": "NO_SIMULATED_AIS_MATCH",
            "label": "No Synthetic AIS Match",
            "color": "#ef4444"
        }
    ]

    for group in marker_groups:

        subset = filtered_df[
            filtered_df["ais_match_status"]
            == group["status"]
        ]

        if subset.empty:
            continue

        hover_text = []

        for _, row in subset.iterrows():

            contrast = row.get("max_contrast_db")

            contrast_text = (
                f"{float(contrast):.2f} dB"
                if pd.notna(contrast)
                else "Unavailable"
            )

            hover_text.append(
                f"<b>Candidate ID: {row['detection_id']}</b>"
                f"<br>Latitude: {row['latitude']:.5f}"
                f"<br>Longitude: {row['longitude']:.5f}"
                f"<br>Maritime Zone: {row['maritime_zone']}"
                f"<br>AIS Status: {row['ais_match_status']}"
                f"<br>Review Priority: {row['review_priority']}"
                f"<br>Maximum Contrast: {contrast_text}"
            )

        fig.add_trace(
            go.Scattermap(
                lat=subset["latitude"].tolist(),
                lon=subset["longitude"].tolist(),
                mode="markers",
                marker={
                    "size": 11,
                    "color": group["color"],
                    "opacity": 0.85
                },
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                name=group["label"]
            )
        )

    fig.update_layout(
        map={
            "style": "open-street-map",
            "center": {
                "lat": (SOUTH + NORTH) / 2,
                "lon": (WEST + EAST) / 2
            },
            "zoom": 9
        },
        height=610,
        margin={
            "l": 0,
            "r": 0,
            "t": 0,
            "b": 0
        },
        legend={
            "orientation": "h",
            "y": -0.08,
            "x": 0
        },
        paper_bgcolor="#ffffff"
    )

    return fig


# ============================================================
# 6. PCA EXPLAINED VARIANCE CHART
# ============================================================

def create_variance_chart():

    df = pd.read_csv(VARIANCE_FILE)

    possible_columns = [
        "explained_variance_ratio",
        "variance_ratio",
        "explained_variance"
    ]

    variance_column = next(
        (
            column for column in possible_columns
            if column in df.columns
        ),
        None
    )

    if variance_column is None:

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if not numeric_columns:
            raise ValueError(
                "No numeric PCA variance data found"
            )

        variance_column = numeric_columns[0]

    values = pd.to_numeric(
        df[variance_column],
        errors="coerce"
    ).dropna()

    if values.max() <= 1.0:
        values = values * 100

    components = [
        f"PC{i + 1}"
        for i in range(len(values))
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=components,
            y=values,
            marker_color="#2563eb",
            text=[
                f"{value:.2f}%"
                for value in values
            ],
            textposition="outside"
        )
    )

    fig.update_layout(
        title="PCA Explained Variance",
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        height=350,
        margin={
            "t": 65,
            "b": 45
        }
    )

    return fig


# ============================================================
# 7. PCA COMPONENT LOADINGS HEATMAP
# ============================================================

def create_loadings_chart():

    df = pd.read_csv(LOADINGS_FILE)

    numeric = df.select_dtypes(
        include="number"
    )

    if numeric.empty:
        raise ValueError(
            "No numeric PCA loading columns found"
        )

    # The five features used in the PCA pipeline.
    feature_names = [
        "VH Backscatter (dB)",
        "VV Backscatter (dB)",
        "Local VV Mean",
        "Local VV Variance",
        "VV - VH Difference (dB)"
    ]

    # Use actual feature names when the number
    # of rows matches the original PCA feature count.
    if len(df) == len(feature_names):
        labels = feature_names

    else:
        labels = [
            f"Feature {i + 1}"
            for i in range(len(df))
        ]

    fig = go.Figure(
        go.Heatmap(
            z=numeric.to_numpy(),
            x=numeric.columns.tolist(),
            y=labels,
            colorscale="RdBu",
            zmid=0,
            colorbar={
                "title": "Loading"
            }
        )
    )

    fig.update_layout(
        title="PCA Component Loadings",
        height=350,
        margin={
            "t": 65,
            "l": 150,
            "b": 45
        }
    )

    return fig


# ============================================================
# 8. CREATE DASH APPLICATION
# ============================================================

app = Dash(__name__)

app.title = "Maritime Surveillance Dashboard"

PAGE_STYLE = {
    "fontFamily": "Arial, sans-serif",
    "backgroundColor": "#f1f5f9",
    "minHeight": "100vh",
    "padding": "25px",
    "color": "#0f172a"
}

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "padding": "18px",
    "borderRadius": "12px",
    "boxShadow": "0 2px 10px rgba(0,0,0,0.06)"
}


# ============================================================
# 9. DROPDOWN OPTIONS
# ============================================================

zone_options = [
    {
        "label": zone,
        "value": zone
    }
    for zone in sorted(
        detections["maritime_zone"].dropna().unique()
    )
]

status_options = [
    {
        "label": "Synthetic AIS Match",
        "value": "SIMULATED_AIS_MATCH"
    },
    {
        "label": "No Synthetic AIS Match",
        "value": "NO_SIMULATED_AIS_MATCH"
    }
]


# ============================================================
# 10. DASHBOARD LAYOUT
# ============================================================

app.layout = html.Div(
    style=PAGE_STYLE,
    children=[

        html.H1(
            "Maritime Surveillance Dashboard"
        ),

        html.P(
            "PCA-Based Sentinel-1 SAR Vessel Candidate Detection "
            "| Mumbai Offshore | 30 September 2026"
        ),

        html.Div(
            "DEMONSTRATION MODE: AIS data is synthetic. "
            "SAR candidates are unverified. "
            "Review categories do not indicate illegal activity.",
            style={
                "padding": "12px",
                "backgroundColor": "#fef3c7",
                "borderRadius": "8px",
                "marginBottom": "20px",
                "fontWeight": "bold",
                "fontSize": "13px"
            }
        ),

        # Summary cards.
        html.Div(
            id="summary-cards",
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "15px",
                "marginBottom": "20px"
            }
        ),

        # Interactive filters.
        html.Div(
            style={
                **CARD_STYLE,
                "marginBottom": "20px"
            },
            children=[

                html.H3("Filter Satellite Candidates"),

                html.Label("Maritime Zone"),

                dcc.Dropdown(
                    id="zone-filter",
                    options=zone_options,
                    multi=True,
                    placeholder="All maritime zones"
                ),

                html.Br(),

                html.Label("AIS Matching Status"),

                dcc.Dropdown(
                    id="ais-filter",
                    options=status_options,
                    multi=True,
                    placeholder="All matching statuses"
                )
            ]
        ),

        # Interactive maritime map.
        html.Div(
            style={
                **CARD_STYLE,
                "marginBottom": "20px"
            },
            children=[

                html.H2("Maritime Detection Map"),

                dcc.Graph(
                    id="maritime-map",
                    figure=create_map(detections)
                )
            ]
        ),

        # PCA visualizations.
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns":
                    "repeat(auto-fit, minmax(350px, 1fr))",
                "gap": "20px",
                "marginBottom": "20px"
            },
            children=[

                html.Div(
                    style=CARD_STYLE,
                    children=[
                        dcc.Graph(
                            figure=create_variance_chart()
                        )
                    ]
                ),

                html.Div(
                    style=CARD_STYLE,
                    children=[
                        dcc.Graph(
                            figure=create_loadings_chart()
                        )
                    ]
                )
            ]
        ),

        html.P(
            "Source: Sentinel-1 SAR imagery. "
            "AIS data: synthetic demonstration dataset.",
            style={
                "textAlign": "center",
                "color": "#64748b",
                "fontSize": "12px"
            }
        )
    ]
)


# ============================================================
# 11. INTERACTIVE CALLBACK
# ============================================================

@app.callback(
    Output("maritime-map", "figure"),
    Output("summary-cards", "children"),
    Input("zone-filter", "value"),
    Input("ais-filter", "value")
)
def update_dashboard(selected_zones, selected_statuses):

    filtered = detections.copy()

    if selected_zones:
        filtered = filtered[
            filtered["maritime_zone"].isin(
                selected_zones
            )
        ]

    if selected_statuses:
        filtered = filtered[
            filtered["ais_match_status"].isin(
                selected_statuses
            )
        ]

    total = len(filtered)

    matched = (
        filtered["ais_match_status"]
        == "SIMULATED_AIS_MATCH"
    ).sum()

    unmatched = total - matched

    contiguous = (
        filtered["maritime_zone"]
        == "Contiguous Zone"
    ).sum()

    eez = (
        filtered["maritime_zone"]
        == "Exclusive Economic Zone"
    ).sum()

    metrics = [
        ("SAR Candidates", total),
        ("Synthetic AIS Matches", matched),
        ("Unmatched Candidates", unmatched),
        ("Contiguous Zone", contiguous),
        ("EEZ", eez)
    ]

    cards = []

    for label, value in metrics:

        cards.append(
            html.Div(
                style=CARD_STYLE,
                children=[

                    html.H2(
                        str(value),
                        style={
                            "fontSize": "30px",
                            "margin": "0",
                            "color": "#2563eb"
                        }
                    ),

                    html.P(
                        label,
                        style={
                            "color": "#475569",
                            "marginBottom": "0"
                        }
                    )
                ]
            )
        )

    return create_map(filtered), cards


# ============================================================
# 12. RUN THE DASHBOARD
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, port=8051)
