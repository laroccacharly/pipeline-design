import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import itertools 

from .data import load_canada_cities_df
from .edmonton import compute_distance_from_edmonton, filter_closest_to_edmonton
from .distance_matrix import get_distance


def create_edge_df(node_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame representing connections (edges) between all pairs of cities
       in the input node_df using itertools."""
    edges = []
    num_nodes = len(node_df)

    # Use itertools.combinations to generate unique pairs of indices
    for i, j in itertools.combinations(range(num_nodes), 2):
        start_city_row = node_df.iloc[i]
        end_city_row = node_df.iloc[j]

        # Use get_distance for accurate distance calculation
        distance = get_distance(start_city_row['id'], end_city_row['id'])

        mid_lat = (start_city_row['lat'] + end_city_row['lat']) / 2
        mid_lon = (start_city_row['lon'] + end_city_row['lon']) / 2
        hover_text = f"{start_city_row['city']} <-> {end_city_row['city']}: {distance:.2f} km"

        edges.append({
            'start_city': start_city_row['city'],
            'end_city': end_city_row['city'],
            'start_lat': start_city_row['lat'],
            'start_lon': start_city_row['lon'],
            'end_lat': end_city_row['lat'],
            'end_lon': end_city_row['lon'],
            'mid_lat': mid_lat,
            'mid_lon': mid_lon,
            'hover_text': hover_text,
            'distance': distance,
            'selected': np.random.randint(0, 2) 
        })

    edge_df = pd.DataFrame(edges)
    if not edge_df.empty:
        edge_df['selected'] = edge_df['selected'].astype(bool)
    print(f"Created edge DataFrame with {len(edge_df)} edges between {num_nodes} cities using itertools. Used get_distance.")
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


