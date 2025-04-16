import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from .data import load_canada_cities_df
import numpy as np 
from .haversine import haversine 
from pydantic import BaseModel
from typing import List

class Arc(BaseModel):
    start_city: str
    end_city: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    mid_lat: float
    mid_lon: float
    hover_text: str

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

def create_arcs(df: pd.DataFrame, num_target_cities: int = 5) -> List[Arc]:
    """Creates a list of Arc objects representing connections from the closest city (Edmonton) 
       to the next `num_target_cities - 1` closest cities."""
    arcs = []

    num_connections_to_make = min(num_target_cities, len(df)) # Ensure we don't go out of bounds
    edmonton = df.iloc[0]

    # Connect Edmonton (index 0) to the next num_connections_to_make - 1 cities
    for i in range(1, num_connections_to_make):
        target_city = df.iloc[i]

        # Use pre-computed distance from Edmonton
        distance = target_city['distance_from_edmonton']

        mid_lat = (edmonton['lat'] + target_city['lat']) / 2
        mid_lon = (edmonton['lon'] + target_city['lon']) / 2
        # Use distance for hover text
        hover_text = f"Distance: {distance:.2f} km"

        arc = Arc(
            start_city=edmonton['city'],
            end_city=target_city['city'],
            start_lat=edmonton['lat'],
            start_lon=edmonton['lon'],
            end_lat=target_city['lat'],
            end_lon=target_city['lon'],
            mid_lat=mid_lat,
            mid_lon=mid_lon,
            hover_text=hover_text
        )
        arcs.append(arc)

    print(f"Created {len(arcs)} arcs originating from Edmonton.")
    return arcs

def create_canada_ui():
    df = load_canada_cities_df()
    df = compute_distance_from_edmonton(df)
    # Filter to top N including Edmonton. Ensure n is large enough for desired connections.
    num_cities_for_arcs = 5 # Edmonton + 4 others
    df = filter_closest_to_edmonton(df, n=num_cities_for_arcs)

    st.title("Canadian Cities Map with Routes (Plotly)")

    # --- Create Arcs --- Connect Edmonton to the next N-1 closest
    arcs = create_arcs(df, num_target_cities=num_cities_for_arcs)
    # -------------------

    # --- Prepare data for Plotly traces from Arcs ---
    line_lats = []
    line_lons = []
    mid_lats = []
    mid_lons = []
    arc_hover_texts = []

    for arc in arcs:
        # Coordinates for the line segment
        line_lats.extend([arc.start_lat, arc.end_lat, None]) # None separates lines
        line_lons.extend([arc.start_lon, arc.end_lon, None])

        # Midpoint for hover label placement
        mid_lats.append(arc.mid_lat)
        mid_lons.append(arc.mid_lon)

        # Hover text for the arc
        arc_hover_texts.append(arc.hover_text)
    # --------------------------------------------------

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

    # Add City Markers Trace - Use the filtered DF for markers as well
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=df['lat'], 
        lon=df['lon'], 
        marker=dict(size=5, color="blue"),
        name="Cities",
        hoverinfo='text',
        hovertext=df['city'] 
    ))

    # Update Layout
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=9,
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


