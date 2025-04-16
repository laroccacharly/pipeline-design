import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from .data import load_canada_cities_df
import numpy as np 
from .haversine import haversine 

def compute_distance_from_edmonton(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the distance of each city from Edmonton using the Haversine formula."""
    try:
        edmonton = df[df['city'] == 'Edmonton'].iloc[0]
        edmonton_lat = edmonton['lat']
        edmonton_lon = edmonton['lon']
    except IndexError:
        st.error("Edmonton not found in the city data. Cannot calculate distances.")
        # Return df unchanged or handle as appropriate
        df['distance_from_edmonton'] = np.nan
        return df

    # Calculate distances using the haversine function
    df['distance_from_edmonton'] = haversine(
        edmonton_lat, edmonton_lon,
        df['lat'].values, df['lon'].values
    )
    print("Calculated distances from Edmonton.")
    return df

def filter_closest_to_edmonton(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    n = min(n, len(df))
    closest_df = df.sort_values(by='distance_from_edmonton').head(n)
    print(f"Filtered to {len(closest_df)} cities closest to Edmonton.")
    return closest_df

def create_canada_ui():
    df = load_canada_cities_df()
    df = compute_distance_from_edmonton(df)
    df = filter_closest_to_edmonton(df, n=10)

    # Column renaming now happens in src/data.py

    st.title("Canadian Cities Map with Routes (Plotly)")

    # --- Define Connections (Example: First 5 cities sequentially) ---
    num_connections = 5
    if len(df) < num_connections:
        st.warning(f"Not enough cities ({len(df)}) to draw {num_connections-1} connections.")
        connections = []
    else:
        # Pairs of indices to connect (e.g., (0, 1), (1, 2), ...)
        connections = list(zip(range(num_connections - 1), range(1, num_connections)))

    line_lats = []
    line_lons = []
    mid_lats = []
    mid_lons = []
    arc_hover_texts = []

    for i, j in connections:
        city1 = df.iloc[i]
        city2 = df.iloc[j]

        # Coordinates for the line segment
        line_lats.extend([city1['lat'], city2['lat'], None]) # None separates lines
        line_lons.extend([city1['lon'], city2['lon'], None])

        # Calculate midpoint for hover label placement
        mid_lat = (city1['lat'] + city2['lat']) / 2
        mid_lon = (city1['lon'] + city2['lon']) / 2
        mid_lats.append(mid_lat)
        mid_lons.append(mid_lon)

        # Define hover text for the arc
        arc_hover_texts.append(f"Route: {city1['city']} -> {city2['city']}")
    # --------------------------------------------------------------

    fig = go.Figure()

    # Add Lines (Arcs) Trace
    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lat=line_lats,
        lon=line_lons,
        line=dict(width=1, color="red"),
        name="Routes",
        hoverinfo='none' # Hover handled by midpoint markers
    ))

    # Add Invisible Midpoint Markers for Arc Hover Info
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=mid_lats,
        lon=mid_lons,
        marker=dict(size=10, color="rgba(255,0,0,0)"), # Slightly larger, fully transparent markers
        name="Route Info",
        hoverinfo='skip', # Skip default hover info generation
        hovertemplate=[f"{text}<extra></extra>" for text in arc_hover_texts] # Use hovertemplate
    ))

    # Add City Markers Trace
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=df['lat'],
        lon=df['lon'],
        marker=dict(size=5, color="blue"),
        name="Cities",
        hoverinfo='text',
        hovertext=df['city'] # Show city name on hover
    ))


    # Update Layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=8,
        mapbox_center_lat = 53.5344, # Edmonton Lat
        mapbox_center_lon = -113.4903, # Edmonton Lon
        margin={"r":0,"t":30,"l":0,"b":0}, # Added top margin for title
        height=600,
        showlegend=True # Show legend for Cities/Routes
    )

    # Explicitly enable scroll zoom in the Plotly config
    config = {'scrollZoom': True}
    st.plotly_chart(fig, use_container_width=True, config=config)


if __name__ == "__main__":
    create_canada_ui()


