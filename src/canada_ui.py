import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import itertools 

from .data import load_canada_cities_df
from .edmonton import compute_distance_from_edmonton, filter_closest_to_edmonton
from .distance_matrix import get_distance
from .mst import solve_mst
from .demand import generate_demand
from .solve_flow_problem import solve_flow_problem

def create_edge_df(node_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame representing connections (edges) between all pairs of cities
       in the input node_df using itertools."""
    edges = []
    num_nodes = len(node_df)

    # Use itertools.combinations to generate unique pairs of indices
    for i, j in itertools.combinations(range(num_nodes), 2):
        source_city_row = node_df.iloc[i]
        target_city_row = node_df.iloc[j]

        # Use get_distance for accurate distance calculation
        distance = get_distance(source_city_row['id'], target_city_row['id'])

        mid_lat = (source_city_row['lat'] + target_city_row['lat']) / 2
        mid_lon = (source_city_row['lon'] + target_city_row['lon']) / 2
        hover_text = f"{source_city_row['city']} <-> {target_city_row['city']}: {distance:.2f} km"

        edges.append({
            'source': source_city_row['id'],
            'target': target_city_row['id'],
            'source_lat': source_city_row['lat'],
            'source_lon': source_city_row['lon'],
            'target_lat': target_city_row['lat'],
            'target_lon': target_city_row['lon'],
            'mid_lat': mid_lat,
            'mid_lon': mid_lon,
            'hover_text': hover_text,
            'distance': distance,
            'selected': 0, 
        })

    edge_df = pd.DataFrame(edges)
    edge_df['selected'] = edge_df['selected'].astype(bool)

    print(f"Created edge DataFrame with {len(edge_df)} edges between {num_nodes} cities using itertools.")
    return edge_df

def get_graph_data(max_node_count: int = 10):
    full_node_df = load_canada_cities_df()
    node_df = compute_distance_from_edmonton(full_node_df)
    max_node_count = max(2, max_node_count)
    node_df = filter_closest_to_edmonton(node_df, n=max_node_count)
    node_df = generate_demand(node_df)
    edge_df = create_edge_df(node_df)
    node_df, edge_df = solve_mst(node_df, edge_df)
    edge_df = edge_df[edge_df['selected']]
    node_df, edge_df = solve_flow_problem(node_df, edge_df)
    return node_df, edge_df

def create_canada_ui():
    st.title("Canadian Cities Map with Routes")

    input_num_cities = st.number_input(
        "Select number of cities to display (closest to Edmonton):",
        min_value=2,  # Need at least 2 cities for an edge
        max_value=30,
        value="min", 
        step=1,
        key="city_input" # Assign a key for potential future reference
    )
    update_button = st.button("Run")

    if 'node_df' not in st.session_state or update_button:
        with st.spinner("Computing routes..."):
            st.session_state.node_df, st.session_state.edge_df = get_graph_data(
                input_num_cities
            )

    node_df = st.session_state.node_df
    edge_df = st.session_state.edge_df

    edge_df = edge_df[edge_df['selected']].copy()

    line_lats = []
    line_lons = []
    for _, edge in edge_df.iterrows():
        line_lats.extend([edge['source_lat'], edge['target_lat'], None])
        line_lons.extend([edge['source_lon'], edge['target_lon'], None])

    mid_lats = edge_df['mid_lat'].tolist()
    mid_lons = edge_df['mid_lon'].tolist()
    arc_hover_texts = edge_df.apply(lambda row: f"{row['hover_text']}<br>Flow: {row['flow']:.2f}", axis=1).tolist()

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
        lat=node_df[node_df['demand_met']]['lat'],
        lon=node_df[node_df['demand_met']]['lon'],
        marker=dict(size=5, color="green"),
        name="Cities (Demand Met)",
        hoverinfo='text',
        hovertext=node_df[node_df['demand_met']].apply(lambda row: f"{row['city']}<br>Demand: {row['demand']:.2f}<br>Served: {row['actual_demand_served']:.2f}", axis=1)
    ))

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=node_df[~node_df['demand_met']]['lat'],
        lon=node_df[~node_df['demand_met']]['lon'],
        marker=dict(size=5, color="red"),
        name="Cities (Demand Unmet)",
        hoverinfo='text',
        hovertext=node_df[~node_df['demand_met']].apply(lambda row: f"{row['city']}<br>Demand: {row['demand']:.2f}<br>Served: {row['actual_demand_served']:.2f}", axis=1)
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=8,
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


