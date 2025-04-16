import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from .data import load_canada_cities_df
from .distance_matrix import get_distance

def compute_distance_from_edmonton(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the distance of each city from Edmonton using the precomputed distance matrix."""
    if 'id' not in df.columns:
        st.error("City data must contain an 'id' column.")
        df['distance_from_edmonton'] = np.nan
        return df

    try:
        # Find Edmonton's ID
        edmonton_id = df.loc[df['city'] == 'Edmonton', 'id'].iloc[0]
    except IndexError:
        st.error("Edmonton not found in the city data or does not have an ID. Cannot calculate distances.")
        df['distance_from_edmonton'] = np.nan
        return df

    # Apply get_distance using Edmonton's ID and each city's ID
    df['distance_from_edmonton'] = df['id'].apply(
        lambda other_id: get_distance(edmonton_id, other_id)
    )
    print(f"Looked up distances from Edmonton (ID: {edmonton_id}).")
    return df

def filter_closest_to_edmonton(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    n = min(n, len(df))
    closest_df = df.sort_values(by='distance_from_edmonton').head(n)
    print(f"Filtered to {len(closest_df)} cities closest to Edmonton.")
    return closest_df

def create_edge_df(node_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame representing connections (edges) from the first city (Edmonton)
       to all other cities in the input node_df."""
    edges = []

    edmonton = node_df.iloc[0]

    for i in range(1, len(node_df)):
        target_city = node_df.iloc[i]

        distance = target_city['distance_from_edmonton']

        mid_lat = (edmonton['lat'] + target_city['lat']) / 2
        mid_lon = (edmonton['lon'] + target_city['lon']) / 2
        hover_text = f"Distance: {distance:.2f} km"

        edges.append({
            'start_city': edmonton['city'],
            'end_city': target_city['city'],
            'start_lat': edmonton['lat'],
            'start_lon': edmonton['lon'],
            'end_lat': target_city['lat'],
            'end_lon': target_city['lon'],
            'mid_lat': mid_lat,
            'mid_lon': mid_lon,
            'hover_text': hover_text,
            'distance': distance,
            'selected': np.random.randint(0, 2)
        })

    edge_df = pd.DataFrame(edges)
    edge_df['selected'] = edge_df['selected'].astype(bool)
    print(f"Created edge DataFrame with {len(edge_df)} edges originating from {edmonton['city']}. Added 'selected' attribute.")
    return edge_df

def create_canada_ui():
    node_df = load_canada_cities_df()
    node_df = compute_distance_from_edmonton(node_df)
    num_cities_display = 10
    node_df = filter_closest_to_edmonton(node_df, n=num_cities_display)

    edge_df = create_edge_df(node_df)

    selected_edge_df = edge_df[edge_df['selected']].copy()
    print(f"Filtered to {len(selected_edge_df)} selected edges.")

    line_lats = []
    line_lons = []
    for _, edge in selected_edge_df.iterrows():
        line_lats.extend([edge['start_lat'], edge['end_lat'], None])
        line_lons.extend([edge['start_lon'], edge['end_lon'], None])

    mid_lats = selected_edge_df['mid_lat'].tolist()
    mid_lons = selected_edge_df['mid_lon'].tolist()
    arc_hover_texts = selected_edge_df['hover_text'].tolist()

    # --- UI ---
    st.title("Canadian Cities Map with Routes (Plotly)")

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lat=line_lats,
        lon=line_lons,
        line=dict(width=1, color="red"),
        name="Routes",
        hoverinfo='none'
    ))

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=mid_lats,
        lon=mid_lons,
        marker=dict(size=10, color="rgba(255,0,0,0)"),
        name="Route Info",
        hoverinfo='skip',
        hovertemplate=[f"{text}<extra></extra>" for text in arc_hover_texts]
    ))

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=node_df['lat'],
        lon=node_df['lon'],
        marker=dict(size=5, color="blue"),
        name="Cities",
        hoverinfo='text',
        hovertext=node_df['city']
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=9,
        mapbox_center_lat = 53.5344,
        mapbox_center_lon = -113.4903,
        margin={"r":0,"t":30,"l":0,"b":0},
        height=600,
        showlegend=True
    )

    config = {'scrollZoom': True}
    st.plotly_chart(fig, use_container_width=True, config=config)


if __name__ == "__main__":
    create_canada_ui()


