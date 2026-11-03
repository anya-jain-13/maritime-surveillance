
from pathlib import Path
import csv
import io
import json
import math

import pandas as pd
import plotly.graph_objects as go

from dash import (
    Dash, dcc, html, Input, Output,
    State, ALL, ctx, no_update
)


# ============================================================
# MARIS — MARITIME INTELLIGENCE
# Complete Dash Application
# ============================================================

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

DETECTIONS_PATH = (
    OUTPUTS / "detections" / "maritime_risk_results.csv"
)
VARIANCE_PATH = (
    OUTPUTS / "pca" / "explained_variance.csv"
)
LOADINGS_PATH = (
    OUTPUTS / "pca" / "pca_loadings.csv"
)
BOUNDARIES_PATH = OUTPUTS / "boundaries"

BG = "#06101d"
PANEL = "#0d2032"
CYAN = "#39d5f5"
GREEN = "#4ee5b2"
ORANGE = "#ffba69"
RED = "#ff667d"
WHITE = "#f1f8ff"
MUTED = "#94aabe"

WEST = 72.17371910726854
SOUTH = 18.800105316807873
EAST = 72.57634401761091
NORTH = 19.00007029905288

CENTER_LAT = (SOUTH + NORTH) / 2
CENTER_LON = (WEST + EAST) / 2


# ============================================================
# LOAD RESEARCH DATA
# ============================================================

if not DETECTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Detection CSV not found:\n{DETECTIONS_PATH}"
    )

detections = pd.read_csv(DETECTIONS_PATH)

required_columns = [
    "detection_id",
    "latitude",
    "longitude",
    "maritime_zone",
    "ais_match_status",
    "review_priority",
]

for column in required_columns:
    if column not in detections.columns:
        raise ValueError(
            f"Missing required CSV column: {column}"
        )

for column in [
    "latitude",
    "longitude",
    "max_contrast_db",
    "area_pixels",
]:
    if column not in detections.columns:
        detections[column] = float("nan")

    detections[column] = pd.to_numeric(
        detections[column],
        errors="coerce"
    )

detections = detections.dropna(
    subset=["latitude", "longitude"]
).copy()

for column in [
    "detection_id",
    "maritime_zone",
    "ais_match_status",
    "review_priority",
]:
    detections[column] = (
        detections[column]
        .fillna("Unknown")
        .astype(str)
    )

detections = detections.reset_index(drop=True)


# ============================================================
# GENERAL HELPERS
# ============================================================

def fmt(value, suffix="", decimals=2):
    if pd.isna(value):
        return "Not available"

    return f"{float(value):.{decimals}f}{suffix}"


def priority_color(priority):
    value = str(priority).strip().lower()

    if "high" in value or "critical" in value:
        return "#ff667d"  # Red

    if "medium" in value or "moderate" in value:
        return "#ffba69"  # Orange

    if "unverified" in value or "unknown" in value:
        return "#a78bfa"  # Purple

    if "low" in value:
        return "#39d5f5"  # Cyan blue

    return "#94aabe"  # Grey for other statuses

def get_candidate(candidate_id):
    if candidate_id is None:
        return None

    rows = detections[
        detections["detection_id"]
        == str(candidate_id)
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


def filter_detections(zones, ais_statuses):
    df = detections.copy()

    if zones:
        df = df[
            df["maritime_zone"].isin(zones)
        ]

    if ais_statuses:
        df = df[
            df["ais_match_status"].isin(ais_statuses)
        ]

    return df


def section_heading(kicker, title, description):
    return html.Div(
        [
            html.Div(
                kicker,
                className="eyebrow"
            ),
            html.H2(title),
            html.P(description),
        ],
        className="section-heading"
    )


def metric_card(label, value, icon, note):
    return html.Div(
        [
            html.Div(
                icon,
                className="metric-symbol"
            ),
            html.Div(
                label,
                className="metric-label"
            ),
            html.Div(
                str(value),
                className="metric-value"
            ),
            html.Div(
                note,
                className="metric-note"
            ),
        ],
        className="metric-card"
    )


def chart_component(chart_id, figure, height):
    return dcc.Graph(
        id=chart_id,
        figure=figure,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False,
        },
        style={
            "width": "100%",
            "height": f"{height}px",
        }
    )


def style_chart(fig, height=390, left=60):
    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            color=MUTED,
            family="Arial",
            size=12
        ),
        margin=dict(
            l=left,
            r=30,
            t=40,
            b=55
        ),
        height=height,
        hoverlabel=dict(
            bgcolor="#18364a",
            font_color=WHITE
        )
    )

    fig.update_xaxes(
        gridcolor="#20394c",
        zeroline=False
    )

    fig.update_yaxes(
        gridcolor="#20394c",
        zeroline=False
    )

    return fig


def empty_chart(message):
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(
            color=MUTED,
            size=14
        )
    )

    return style_chart(fig)


# ============================================================
# LOAD MARITIME BOUNDARIES
# ============================================================

BOUNDARY_CONFIG = {
    "territorial_sea": {
        "name": "Territorial Sea",
        "color": RED
    },
    "contiguous_zone": {
        "name": "Contiguous Zone",
        "color": ORANGE
    },
    "eez": {
        "name": "Exclusive Economic Zone",
        "color": CYAN
    },
}


def load_boundaries():
    result = {}

    for key in BOUNDARY_CONFIG:
        path = (
            BOUNDARIES_PATH / f"{key}.geojson"
        )

        if not path.exists():
            result[key] = []
            continue

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        result[key] = data.get(
            "features", []
        )

    return result


BOUNDARIES = load_boundaries()


def polygon_coordinates(geometry):
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        return [
            polygon[0]
            for polygon in [coordinates]
            if polygon
        ]

    if geometry_type == "MultiPolygon":
        return [
            polygon[0]
            for polygon in coordinates
            if polygon
        ]

    return []


# ============================================================
# INTERACTIVE MARITIME MAP
# ============================================================

def make_map(df, selected_id=None):
    fig = go.Figure()

    for key, config in BOUNDARY_CONFIG.items():
        first = True

        for feature in BOUNDARIES.get(key, []):
            geometry = feature.get(
                "geometry", {}
            )

            for ring in polygon_coordinates(geometry):
                if not ring:
                    continue

                fig.add_trace(
                    go.Scattermap(
                        lon=[
                            point[0]
                            for point in ring
                        ],
                        lat=[
                            point[1]
                            for point in ring
                        ],
                        mode="lines",
                        line=dict(
                            color=config["color"],
                            width=2
                        ),
                        name=config["name"],
                        legendgroup=key,
                        showlegend=first,
                        hoverinfo="skip"
                    )
                )

                first = False

    study_lon = [
        WEST, EAST, EAST, WEST, WEST
    ]

    study_lat = [
        SOUTH, SOUTH, NORTH, NORTH, SOUTH
    ]

    fig.add_trace(
        go.Scattermap(
            lon=study_lon,
            lat=study_lat,
            mode="lines",
            line=dict(
                color=GREEN,
                width=3
            ),
            name="SAR Study Area",
            hoverinfo="skip"
        )
    )

    for priority in sorted(
        df["review_priority"].unique()
    ):
        subset = df[
            df["review_priority"] == priority
        ]

        customdata = subset[
            [
                "detection_id",
                "maritime_zone",
                "ais_match_status",
                "review_priority"
            ]
        ].astype(str).to_numpy()

        fig.add_trace(
            go.Scattermap(
                lat=subset["latitude"].tolist(),
                lon=subset["longitude"].tolist(),
                mode="markers",
                marker=dict(
                    size=14,
                    color=priority_color(priority),
                    opacity=0.95
                ),
                customdata=customdata,
                name=f"{priority} priority",
                hovertemplate=(
                    "<b>SAR Candidate %{customdata[0]}</b>"
                    "<br>Zone: %{customdata[1]}"
                    "<br>Synthetic AIS: %{customdata[2]}"
                    "<br>Priority: %{customdata[3]}"
                    "<br>Latitude: %{lat:.5f}"
                    "<br>Longitude: %{lon:.5f}"
                    "<br>Click to investigate"
                    "<extra></extra>"
                )
            )
        )

    center_lat = CENTER_LAT
    center_lon = CENTER_LON
    zoom = 9

    if selected_id is not None:
        row = get_candidate(selected_id)

        if row is not None:
            center_lat = float(row["latitude"])
            center_lon = float(row["longitude"])
            zoom = 11

            fig.add_trace(
                go.Scattermap(
                    lat=[center_lat],
                    lon=[center_lon],
                    mode="markers",
                    marker=dict(
                        size=25,
                        color="#ffffff"
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(
                lat=center_lat,
                lon=center_lon
            ),
            zoom=zoom
        ),
        height=650,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=65
        ),
        paper_bgcolor=PANEL,
        font=dict(color=WHITE),
        legend=dict(
            orientation="h",
            x=0,
            y=-0.04,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)"
        ),
        clickmode="event+select",
        uirevision=f"maris-map-{selected_id}"
    )

    return fig


# ============================================================
# GEOGRAPHIC RADAR
# ============================================================

def geographic_position(latitude, longitude):
    lat1 = math.radians(CENTER_LAT)
    lat2 = math.radians(float(latitude))

    dlat = lat2 - lat1

    dlon = math.radians(
        float(longitude) - CENTER_LON
    )

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    distance = (
        2 * 6371.0088
        * math.asin(
            min(1, math.sqrt(a))
        )
    )

    y = (
        math.sin(dlon)
        * math.cos(lat2)
    )

    x = (
        math.cos(lat1)
        * math.sin(lat2)
        - math.sin(lat1)
        * math.cos(lat2)
        * math.cos(dlon)
    )

    bearing = (
        math.degrees(
            math.atan2(y, x)
        ) + 360
    ) % 360

    return distance, bearing


def make_radar(df):
    fig = go.Figure()

    if not df.empty:
        radar_df = df.copy()

        positions = [
            geographic_position(
                row.latitude,
                row.longitude
            )
            for row in radar_df.itertuples()
        ]

        radar_df["range_km"] = [
            item[0]
            for item in positions
        ]

        radar_df["bearing_deg"] = [
            item[1]
            for item in positions
        ]

        for priority in sorted(
            radar_df["review_priority"].unique()
        ):
            subset = radar_df[
                radar_df["review_priority"]
                == priority
            ]

            customdata = subset[
                [
                    "detection_id",
                    "maritime_zone",
                    "ais_match_status"
                ]
            ].astype(str).to_numpy()

            fig.add_trace(
                go.Scatterpolar(
                    r=subset["range_km"].tolist(),
                    theta=subset["bearing_deg"].tolist(),
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=priority_color(priority),
                        opacity=0.95,
                        line=dict(
                            color="#d6f9ff",
                            width=0.5
                        )
                    ),
                    customdata=customdata,
                    name=str(priority),
                    hovertemplate=(
                        "<b>Candidate %{customdata[0]}</b>"
                        "<br>Range: %{r:.2f} km"
                        "<br>Bearing: %{theta:.1f}°"
                        "<br>Zone: %{customdata[1]}"
                        "<br>AIS: %{customdata[2]}"
                        "<br>Click to investigate"
                        "<extra></extra>"
                    )
                )
            )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=WHITE,
            size=10
        ),
        height=430,
        margin=dict(
            l=28,
            r=28,
            t=25,
            b=65
        ),
        polar=dict(
            bgcolor="rgba(4,28,39,0.8)",
            radialaxis=dict(
                visible=True,
                gridcolor="#27617a",
                ticksuffix=" km",
                angle=45
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                gridcolor="#27617a"
            )
        ),
        legend=dict(
            orientation="h",
            x=0,
            y=-0.13
        ),
        clickmode="event+select",
        uirevision="maris-radar"
    )

    return fig


# ============================================================
# PCA CHARTS
# ============================================================

def variance_chart():
    if not VARIANCE_PATH.exists():
        return empty_chart(
            "Explained variance CSV not found"
        )

    df = pd.read_csv(VARIANCE_PATH)

    column = next(
        (
            col
            for col in [
                "explained_variance_ratio",
                "variance_ratio",
                "explained_variance"
            ]
            if col in df.columns
        ),
        None
    )

    if column is None:
        numeric = df.select_dtypes(
            include="number"
        )

        if numeric.empty:
            return empty_chart(
                "No numeric variance data"
            )

        column = numeric.columns[0]

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return empty_chart(
            "No variance values available"
        )

    if values.max() <= 1:
        values = values * 100

    labels = [
        f"PC{i + 1}"
        for i in range(len(values))
    ]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values.tolist(),
            marker_color=[
                CYAN if i == 0 else "#287e9d"
                for i in range(len(values))
            ],
            text=[
                f"{value:.1f}%"
                for value in values
            ],
            textposition="outside"
        )
    )

    fig.update_layout(
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance (%)",
        showlegend=False
    )

    return style_chart(fig, 400)


def loadings_chart():
    if not LOADINGS_PATH.exists():
        return empty_chart(
            "PCA loadings CSV not found"
        )

    df = pd.read_csv(LOADINGS_PATH)

    numeric = df.select_dtypes(
        include="number"
    )

    if numeric.empty:
        return empty_chart(
            "No numeric PCA loading data"
        )

    if len(df) == 5:
        labels = [
            "VH Backscatter",
            "VV Backscatter",
            "Local VV Mean",
            "Local VV Variance",
            "VV - VH Difference"
        ]
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
            colorscale=[
                [0, RED],
                [0.5, "#102c41"],
                [1, CYAN]
            ],
            zmid=0,
            colorbar=dict(
                title="Loading"
            )
        )
    )

    fig.update_layout(
        yaxis=dict(
            autorange="reversed"
        )
    )

    return style_chart(
        fig,
        400,
        170
    )


def zone_chart(df):
    if df.empty:
        return empty_chart(
            "No matching detections"
        )

    counts = (
        df["maritime_zone"]
        .value_counts()
    )

    fig = go.Figure(
        go.Bar(
            x=counts.values.tolist(),
            y=counts.index.tolist(),
            orientation="h",
            marker_color=CYAN,
            text=counts.values.tolist(),
            textposition="auto"
        )
    )

    fig.update_layout(
        xaxis_title="SAR Candidates"
    )

    return style_chart(
        fig,
        370,
        160
    )


def priority_chart(df):
    if df.empty:
        return empty_chart(
            "No matching detections"
        )

    counts = (
        df["review_priority"]
        .value_counts()
    )

    fig = go.Figure(
        go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.67,
            marker=dict(
                colors=[
                    priority_color(priority)
                    for priority in counts.index
                ]
            ),
            textinfo="label+percent"
        )
    )

    fig.update_layout(
        showlegend=False
    )

    return style_chart(
        fig,
        370
    )


# ============================================================
# VESSEL REGISTRY CARDS
# ============================================================

def vessel_card(row):
    candidate_id = str(
        row["detection_id"]
    )

    priority = str(
        row["review_priority"]
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "SAR DETECTION",
                        className="eyebrow"
                    ),
                    html.Span(
                        priority.upper(),
                        className="priority-pill",
                        style={
                            "color": priority_color(priority),
                            "borderColor": priority_color(priority)
                        }
                    )
                ],
                className="vessel-card-top"
            ),

            html.Div(
                [
                    html.Div(
                        "⛴",
                        className="vessel-icon"
                    ),
                    html.Div(
                        "ILLUSTRATIVE VESSEL",
                        className="ship-caption"
                    )
                ],
                className="vessel-visual"
            ),

            html.H3(
                f"Candidate {candidate_id}"
            ),

            html.P(
                str(row["maritime_zone"]),
                className="muted"
            ),

            html.Div(
                [
                    html.Span(
                        "Synthetic AIS"
                    ),
                    html.Strong(
                        str(
                            row["ais_match_status"]
                        ).replace("_", " ")
                    )
                ],
                className="vessel-detail-row"
            ),

            html.Div(
                [
                    html.Span(
                        "Position"
                    ),
                    html.Strong(
                        f"{row['latitude']:.4f}, "
                        f"{row['longitude']:.4f}"
                    )
                ],
                className="vessel-detail-row"
            ),

            html.Div(
                [
                    html.Span(
                        "Maximum contrast"
                    ),
                    html.Strong(
                        fmt(
                            row["max_contrast_db"],
                            " dB"
                        )
                    )
                ],
                className="vessel-detail-row"
            ),

            html.Button(
                "INVESTIGATE TARGET ↗",
                id={
                    "type": "investigate",
                    "index": candidate_id
                },
                n_clicks=0,
                className="investigate-button"
            )
        ],
        className="vessel-card"
    )


# ============================================================
# INTELLIGENCE REPORT
# ============================================================

def intelligence_summary(row):
    candidate_id = str(
        row["detection_id"]
    )

    zone = str(
        row["maritime_zone"]
    )

    priority = str(
        row["review_priority"]
    )

    ais = str(
        row["ais_match_status"]
    )

    contrast = fmt(
        row["max_contrast_db"],
        " dB"
    )

    area = fmt(
        row["area_pixels"],
        " pixels",
        0
    )

    if ais == "SIMULATED_AIS_MATCH":
        ais_text = (
            "The synthetic AIS demonstration "
            "indicates a simulated match. "
            "This is not a verified real-world "
            "AIS correlation."
        )
    elif ais == "NO_SIMULATED_AIS_MATCH":
        ais_text = (
            "No simulated AIS match was recorded. "
            "This does not establish suspicious "
            "or illegal activity."
        )
    else:
        ais_text = (
            "The synthetic AIS comparison "
            "status is not confirmed."
        )

    return (
        f"SAR Candidate {candidate_id} was "
        f"identified in the {zone}. "
        f"The maximum recorded SAR contrast "
        f"is {contrast}, and the detection "
        f"footprint covers {area}. "
        f"{ais_text} "
        f"The exploratory review priority "
        f"is {priority}. "
        "This candidate is unverified. "
        "Further satellite imagery inspection "
        "and independent corroboration "
        "would be required to establish "
        "vessel identity or behavior."
    )


def intelligence_stat(label, value, note=""):
    return html.Div(
        [
            html.Div(
                label,
                className="intel-stat-label"
            ),
            html.Div(
                str(value),
                className="intel-stat-value"
            ),
            html.Div(
                note,
                className="intel-stat-note"
            )
        ],
        className="intel-stat"
    )


def investigation_panel(candidate_id):
    row = get_candidate(candidate_id)

    if row is None:
        return html.Div(
            [
                html.Div(
                    "⌖",
                    className="empty-icon"
                ),
                html.H3(
                    "Awaiting target selection"
                ),
                html.P(
                    "Select a detection on the map, "
                    "radar, or vessel registry to "
                    "open its intelligence report."
                )
            ],
            className="investigation-empty"
        )

    cid = str(row["detection_id"])
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    zone = str(row["maritime_zone"])
    priority = str(row["review_priority"])
    ais = str(row["ais_match_status"])

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "MARIS / TARGET INTELLIGENCE REPORT",
                                className="eyebrow"
                            ),
                            html.H3(
                                f"SAR Candidate #{cid}",
                                className="intel-heading"
                            ),
                            html.P(
                                "Mumbai Offshore · Sentinel-1 SAR",
                                className="muted"
                            )
                        ]
                    ),
                    html.Span(
                        "UNVERIFIED DETECTION",
                        className="intel-unverified"
                    )
                ],
                className="intel-head"
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "SAR SOURCE IMAGERY",
                                className="eyebrow"
                            ),
                            html.Div(
                                "◈",
                                className="intel-imagery-icon"
                            ),
                            html.H4(
                                "Satellite crop unavailable"
                            ),
                            html.P(
                                "A georeferenced Sentinel-1 "
                                "crop has not been connected "
                                "to this candidate. "
                                "The measurements below "
                                "come from the research CSV."
                            )
                        ],
                        className="intel-imagery"
                    ),

                    html.Div(
                        [
                            html.Div(
                                "GEOGRAPHIC CLASSIFICATION",
                                className="eyebrow"
                            ),
                            html.H4(zone),

                            html.Div(
                                "REVIEW PRIORITY",
                                className="intel-small-label"
                            ),
                            html.P(priority),

                            html.Div(
                                "SYNTHETIC AIS STATUS",
                                className="intel-small-label"
                            ),
                            html.P(
                                ais.replace("_", " ")
                            ),

                            html.P(
                                "AIS comparison is simulated.",
                                className="muted"
                            )
                        ],
                        className="intel-geography"
                    )
                ],
                className="intel-top-grid"
            ),

            html.Div(
                [
                    intelligence_stat(
                        "LATITUDE",
                        f"{abs(lat):.5f}° "
                        + ("N" if lat >= 0 else "S")
                    ),
                    intelligence_stat(
                        "LONGITUDE",
                        f"{abs(lon):.5f}° "
                        + ("E" if lon >= 0 else "W")
                    ),
                    intelligence_stat(
                        "MAXIMUM SAR CONTRAST",
                        fmt(
                            row["max_contrast_db"],
                            " dB"
                        )
                    ),
                    intelligence_stat(
                        "DETECTION AREA",
                        fmt(
                            row["area_pixels"],
                            " pixels",
                            0
                        )
                    )
                ],
                className="intel-stat-grid"
            ),

            html.Div(
                [
                    html.Div(
                        "AUTOMATED ANALYTICAL SUMMARY",
                        className="eyebrow"
                    ),
                    html.P(
                        intelligence_summary(row)
                    )
                ],
                className="intel-summary"
            ),

            html.Div(
                [
                    html.Strong(
                        "RESEARCH LIMITATIONS"
                    ),
                    html.P(
                        "This is an unverified SAR "
                        "candidate. Synthetic AIS "
                        "data cannot establish vessel "
                        "identity, vessel type, behavior, "
                        "or illegal activity."
                    )
                ],
                className="investigation-warning"
            ),

            html.Div(
                [
                    html.A(
                        "⌖ LOCATE ON MAP ↗",
                        href=(
                            "https://www.google.com/maps/"
                            f"search/?api=1&query={lat},{lon}"
                        ),
                        target="_blank",
                        className="intel-action"
                    ),
                    html.Button(
                        "↓ EXPORT TARGET REPORT",
                        id="export-target-report",
                        n_clicks=0,
                        className=(
                            "intel-action "
                            "intel-action-primary"
                        )
                    )
                ],
                className="intel-actions"
            )
        ],
        className="investigation-details"
    )


# ============================================================
# DASH APP
# ============================================================

app = Dash(
    __name__,
    suppress_callback_exceptions=True
)

server = app.server

app.title = (
    "MARIS | Maritime Intelligence"
)

zone_options = [
    {
        "label": zone,
        "value": zone
    }
    for zone in sorted(
        detections["maritime_zone"].unique()
    )
]

ais_options = [
    {
        "label": status.replace("_", " "),
        "value": status
    }
    for status in sorted(
        detections["ais_match_status"].unique()
    )
]


# ============================================================
# PAGE LAYOUT
# ============================================================

app.layout = html.Div(
    [
        dcc.Store(
            id="selected-candidate"
        ),

        dcc.Download(
            id="target-report-download"
        ),

        # NAVIGATION
        html.Header(
            [
                html.A(
                    [
                        html.Span(
                            "◈",
                            className="brand-icon"
                        ),
                        html.Span(
                            "MARIS",
                            className="brand-name"
                        ),
                        html.Span(
                            "INTELLIGENCE",
                            className="brand-caption"
                        )
                    ],
                    href="#home",
                    className="brand"
                ),

                html.Nav(
                    [
                        html.A(
                            "Overview",
                            href="#home"
                        ),
                        html.A(
                            "Surveillance",
                            href="#surveillance"
                        ),
                        html.A(
                            "Vessels",
                            href="#vessels"
                        ),
                        html.A(
                            "Analytics",
                            href="#analytics"
                        ),
                        html.A(
                            "Methodology",
                            href="#methodology"
                        )
                    ],
                    className="nav-links"
                ),

                html.Div(
                    [
                        html.Span(
                            className="status-dot"
                        ),
                        "SAR RESEARCH MODE"
                    ],
                    className="system-status"
                )
            ],
            className="topbar"
        ),

        # CINEMATIC HERO
        # The ocean animation is automatically
        # inserted by assets/ocean.js.

        html.Section(
            [
                html.Div(
                    className="hero-grid"
                ),

                html.Div(
                    [
                        html.Div(
                            "SATELLITE INTELLIGENCE / ARABIAN SEA",
                            className="eyebrow"
                        ),

                        html.H1(
                            [
                                "THE OCEAN.",
                                html.Br(),
                                html.Span(
                                    "REIMAGINED."
                                )
                            ]
                        ),

                        html.P(
                            "Discover satellite-derived "
                            "maritime patterns through "
                            "Sentinel-1 SAR imagery, "
                            "Principal Component Analysis "
                            "and geospatial vessel-candidate "
                            "intelligence."
                        ),

                        html.Div(
                            [
                                html.A(
                                    "EXPLORE SURVEILLANCE ↗",
                                    href="#surveillance",
                                    className="hero-button"
                                ),

                                html.Span(
                                    "MUMBAI OFFSHORE · 30 SEP 2026",
                                    className="hero-meta"
                                )
                            ],
                            className="hero-actions"
                        )
                    ],
                    className="hero-content"
                ),

                html.Div(
                    [
                        html.Div(
                            "⛴",
                            className="hero-ship-icon"
                        ),
                        html.Div(
                            "SENTINEL-1 / MARITIME INTELLIGENCE",
                            className="ship-caption"
                        )
                    ],
                    className="hero-ship"
                ),

                html.Div(
                    "01 / MARITIME INTELLIGENCE",
                    className="hero-index"
                )
            ],
            id="home",
            className="hero"
        ),

        html.Main(
            [
                html.Div(
                    [
                        html.Strong(
                            "RESEARCH DEMONSTRATION · "
                        ),
                        "SAR detections are unverified. "
                        "AIS matching uses synthetic data. "
                        "Review priority does not establish "
                        "illegal activity."
                    ],
                    className="demo-notice"
                ),

                html.Div(
                    id="metric-grid",
                    className="metric-grid"
                ),

                # SURVEILLANCE
                html.Section(
                    [
                        section_heading(
                            "02 / GEOSPATIAL INTELLIGENCE",
                            "Maritime Surveillance",
                            "Interactive satellite detection "
                            "mapping and geographic "
                            "target investigation."
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            className="status-dot"
                                                        ),
                                                        "SATELLITE DETECTION MAP"
                                                    ],
                                                    className="panel-kicker"
                                                ),
                                                html.Div(
                                                    "ARABIAN SEA / INDIA",
                                                    className="map-location"
                                                )
                                            ],
                                            className="map-panel-heading"
                                        ),

                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "MARITIME ZONE"
                                                        ),
                                                        dcc.Dropdown(
                                                            id="zone-filter",
                                                            options=zone_options,
                                                            multi=True,
                                                            placeholder="All zones"
                                                        )
                                                    ]
                                                ),

                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "SYNTHETIC AIS STATUS"
                                                        ),
                                                        dcc.Dropdown(
                                                            id="ais-filter",
                                                            options=ais_options,
                                                            multi=True,
                                                            placeholder="All statuses"
                                                        )
                                                    ]
                                                )
                                            ],
                                            className="filter-grid"
                                        ),

                                        chart_component(
                                            "maritime-map",
                                            make_map(detections),
                                            650
                                        ),

                                        html.Div(
                                            "Click a colored marker to "
                                            "inspect a SAR candidate.",
                                            className="map-footnote"
                                        )
                                    ],
                                    className="map-panel"
                                ),

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    "GEOGRAPHIC RADAR REPLAY",
                                                    className="eyebrow"
                                                ),

                                                html.H3(
                                                    "SAR Target Radar"
                                                ),

                                                html.Div(
                                                    [
                                                        html.Div(
                                                            className="radar-sweep-overlay"
                                                        ),
                                                        chart_component(
                                                            "detection-radar",
                                                            make_radar(detections),
                                                            430
                                                        )
                                                    ],
                                                    className="radar-stage"
                                                ),

                                                html.Div(
                                                    id="radar-count",
                                                    className="radar-count"
                                                ),

                                                html.P(
                                                    "Candidate positions "
                                                    "are plotted using "
                                                    "geographic distance "
                                                    "and bearing. The sweep "
                                                    "is illustrative, not "
                                                    "live radar tracking.",
                                                    className="radar-disclaimer"
                                                )
                                            ],
                                            className="radar-panel"
                                        ),

                                        html.Div(
                                            [
                                                html.Div(
                                                    "TARGET INVESTIGATION",
                                                    className="eyebrow"
                                                ),
                                                html.Div(
                                                    id="investigation-content",
                                                    children=investigation_panel(None)
                                                )
                                            ],
                                            className="investigation-panel"
                                        )
                                    ],
                                    className="surveillance-side"
                                )
                            ],
                            className="surveillance-layout"
                        )
                    ],
                    id="surveillance",
                    className="content-section"
                ),

                # VESSEL REGISTRY
                html.Section(
                    [
                        section_heading(
                            "03 / TARGET REGISTRY",
                            "Vessel Candidates",
                            "Explore SAR signatures "
                            "and investigate individual "
                            "candidate detections."
                        ),

                        html.Div(
                            id="vessel-cards",
                            className="vessel-grid"
                        )
                    ],
                    id="vessels",
                    className="content-section"
                ),

                # ANALYTICS
                html.Section(
                    [
                        section_heading(
                            "04 / ANALYTICAL INTELLIGENCE",
                            "PCA & Detection Analytics",
                            "Principal component analysis "
                            "and exploratory detection "
                            "statistics."
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3(
                                            "Explained Variance"
                                        ),
                                        html.P(
                                            "Variance captured by "
                                            "each principal component.",
                                            className="muted"
                                        ),
                                        chart_component(
                                            "variance-chart",
                                            variance_chart(),
                                            400
                                        )
                                    ],
                                    className="chart-panel"
                                ),

                                html.Div(
                                    [
                                        html.H3(
                                            "PCA Feature Loadings"
                                        ),
                                        html.P(
                                            "Contribution of SAR "
                                            "features to principal "
                                            "components.",
                                            className="muted"
                                        ),
                                        chart_component(
                                            "loadings-chart",
                                            loadings_chart(),
                                            400
                                        )
                                    ],
                                    className="chart-panel"
                                ),

                                html.Div(
                                    [
                                        html.H3(
                                            "Maritime Zone Distribution"
                                        ),
                                        html.P(
                                            "Candidate counts by "
                                            "maritime zone.",
                                            className="muted"
                                        ),
                                        chart_component(
                                            "zone-chart",
                                            zone_chart(detections),
                                            370
                                        )
                                    ],
                                    className="chart-panel"
                                ),

                                html.Div(
                                    [
                                        html.H3(
                                            "Review Priority"
                                        ),
                                        html.P(
                                            "Exploratory priority "
                                            "distribution.",
                                            className="muted"
                                        ),
                                        chart_component(
                                            "priority-chart",
                                            priority_chart(detections),
                                            370
                                        )
                                    ],
                                    className="chart-panel"
                                )
                            ],
                            className="analytics-grid"
                        )
                    ],
                    id="analytics",
                    className="content-section"
                ),

                # METHODOLOGY
                html.Section(
                    [
                        section_heading(
                            "05 / RESEARCH PIPELINE",
                            "From Satellite to Insight",
                            "The five-stage MARIS "
                            "research workflow."
                        ),

                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            f"0{i + 1}",
                                            className="step-number"
                                        ),
                                        html.H3(title),
                                        html.P(description)
                                    ],
                                    className="method-step"
                                )
                                for i, (
                                    title,
                                    description
                                ) in enumerate(
                                    [
                                        (
                                            "SAR Acquisition",
                                            "Sentinel-1 imagery "
                                            "from the Mumbai "
                                            "offshore study area."
                                        ),
                                        (
                                            "Feature Extraction",
                                            "Radar backscatter "
                                            "and local statistical "
                                            "features."
                                        ),
                                        (
                                            "PCA Analysis",
                                            "Dimensionality reduction "
                                            "and feature contribution "
                                            "analysis."
                                        ),
                                        (
                                            "Candidate Detection",
                                            "Identification of "
                                            "vessel-like SAR "
                                            "signatures."
                                        ),
                                        (
                                            "Synthetic AIS Comparison",
                                            "Research demonstration "
                                            "using simulated "
                                            "AIS records."
                                        )
                                    ]
                                )
                            ],
                            className="method-grid"
                        )
                    ],
                    id="methodology",
                    className="content-section"
                ),

                html.Footer(
                    [
                        html.Strong(
                            "MARIS / MARITIME INTELLIGENCE"
                        ),
                        html.P(
                            "Sentinel-1 SAR research "
                            "demonstration · Synthetic AIS "
                            "· Unverified detections"
                        )
                    ],
                    className="footer"
                )
            ],
            className="main-content"
        )
    ],
    className="maris-app"
)


# ============================================================
# DASHBOARD CALLBACK
# ============================================================

@app.callback(
    Output("maritime-map", "figure"),
    Output("detection-radar", "figure"),
    Output("radar-count", "children"),
    Output("metric-grid", "children"),
    Output("vessel-cards", "children"),
    Output("zone-chart", "figure"),
    Output("priority-chart", "figure"),
    Input("zone-filter", "value"),
    Input("ais-filter", "value")
)
def update_dashboard(zones, statuses):
    df = filter_detections(
        zones,
        statuses
    )

    total = len(df)

    matched = int(
        (
            df["ais_match_status"]
            == "SIMULATED_AIS_MATCH"
        ).sum()
    )

    unmatched = int(
        (
            df["ais_match_status"]
            == "NO_SIMULATED_AIS_MATCH"
        ).sum()
    )

    high_priority = int(
        df["review_priority"].str.contains(
            "high|critical|urgent",
            case=False,
            regex=True
        ).sum()
    )

    metrics = [
        metric_card(
            "SAR CANDIDATES",
            total,
            "⛴",
            "Unverified detections"
        ),
        metric_card(
            "SYNTHETIC AIS MATCHES",
            matched,
            "◈",
            "Simulated AIS comparison"
        ),
        metric_card(
            "NO SYNTHETIC MATCH",
            unmatched,
            "⌖",
            "May require further review"
        ),
        metric_card(
            "HIGH REVIEW PRIORITY",
            high_priority,
            "⚑",
            "Not evidence of illegality"
        )
    ]

    cards = [
        vessel_card(row)
        for _, row in df.head(12).iterrows()
    ]

    if not cards:
        cards = [
            html.P(
                "No detections match your filters.",
                className="muted"
            )
        ]

    return (
        make_map(df),
        make_radar(df),
        f"{total} SAR candidates plotted",
        metrics,
        cards,
        zone_chart(df),
        priority_chart(df)
    )


# ============================================================
# TARGET SELECTION
# NO AUTOMATIC SCROLLING
# ============================================================

@app.callback(
    Output("selected-candidate", "data"),
    Input("maritime-map", "clickData"),
    Input("detection-radar", "clickData"),
    Input(
        {
            "type": "investigate",
            "index": ALL
        },
        "n_clicks"
    ),
    State("selected-candidate", "data"),
    prevent_initial_call=True
)
def select_candidate(
    map_click,
    radar_click,
    buttons,
    current
):
    trigger = ctx.triggered_id

    if trigger in [
        "maritime-map",
        "detection-radar"
    ]:
        click = (
            map_click
            if trigger == "maritime-map"
            else radar_click
        )

        if click and click.get("points"):
            customdata = (
                click["points"][0]
                .get("customdata")
            )

            if (
                customdata is not None
                and len(customdata) > 0
            ):
                return str(customdata[0])

    if isinstance(trigger, dict):
        if trigger.get("type") == "investigate":
            return str(
                trigger["index"]
            )

    return no_update


# ============================================================
# SHOW TARGET INVESTIGATION
# ============================================================

@app.callback(
    Output(
        "investigation-content",
        "children"
    ),
    Input(
        "selected-candidate",
        "data"
    )
)
def show_investigation(candidate_id):
    return investigation_panel(candidate_id)


# ============================================================
# EXPORT TARGET REPORT
# ============================================================

@app.callback(
    Output(
        "target-report-download",
        "data"
    ),
    Input(
        "export-target-report",
        "n_clicks"
    ),
    State(
        "selected-candidate",
        "data"
    ),
    prevent_initial_call=True
)
def export_target_report(
    n_clicks,
    candidate_id
):
    if not n_clicks:
        return no_update

    row = get_candidate(candidate_id)

    if row is None:
        return no_update

    fields = [
        (
            "Candidate ID",
            str(row["detection_id"])
        ),
        (
            "Latitude",
            fmt(row["latitude"], "", 6)
        ),
        (
            "Longitude",
            fmt(row["longitude"], "", 6)
        ),
        (
            "Maritime Zone",
            str(row["maritime_zone"])
        ),
        (
            "Review Priority",
            str(row["review_priority"])
        ),
        (
            "Synthetic AIS Status",
            str(row["ais_match_status"])
        ),
        (
            "Maximum SAR Contrast",
            fmt(
                row["max_contrast_db"],
                " dB"
            )
        ),
        (
            "Detection Area",
            fmt(
                row["area_pixels"],
                " pixels",
                0
            )
        ),
        (
            "Analytical Summary",
            intelligence_summary(row)
        ),
        (
            "Research Limitations",
            (
                "Unverified SAR candidate. "
                "Synthetic AIS comparison. "
                "No confirmed vessel identity, "
                "behavior, or illegal activity."
            )
        )
    ]

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow(
        [
            "MARIS TARGET INTELLIGENCE REPORT",
            ""
        ]
    )

    writer.writerow(
        ["Field", "Value"]
    )

    writer.writerows(fields)

    safe_id = "".join(
        character
        for character in str(candidate_id)
        if character.isalnum()
        or character in "-_"
    )

    return {
        "content": output.getvalue(),
        "filename": (
            f"MARIS_Candidate_{safe_id}.csv"
        ),
        "type": "text/csv"
    }


# ============================================================
# IMPORTANT:
# AUTOMATIC SCROLLING IS COMPLETELY DISABLED.
#
# There is intentionally NO scrollIntoView()
# callback in this application.
#
# The website remains on the cinematic homepage
# when it first opens.
# ============================================================


# ============================================================
# START MARIS
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=False,
        port=8051
    )
