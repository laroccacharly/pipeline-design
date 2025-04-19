import streamlit as st
import plotly.graph_objects as go

from .data import load_canada_cities_df
from .edmonton import compute_distance_from_edmonton, filter_closest_to_edmonton
from .mst import solve_mst
from .demand import generate_demand
from .solve_flow_problem_grb import solve_flow_problem_grb
from .solve_flow_problem_highs import solve_flow_problem_highs
from .config import Config
from .create_edge_df import create_edge_df

def get_graph_data(config: Config):
    full_node_df = load_canada_cities_df()
    node_df = compute_distance_from_edmonton(full_node_df)
    max_node_count = max(2, config.max_node_count)
    node_df = filter_closest_to_edmonton(node_df, n=max_node_count)
    node_df = generate_demand(node_df, config.average_capacity_per_node, config.average_demand_per_node)
    edge_df = create_edge_df(node_df)
    node_df, edge_df = solve_mst(node_df, edge_df)
    edge_df = edge_df[edge_df['selected']]
    original_edge_df = edge_df.copy()
    original_node_df = node_df.copy()
    grb_node_df, grb_edge_df, grb_metrics = solve_flow_problem_grb(original_node_df, original_edge_df)
    highs_node_df, highs_edge_df, highs_metrics = solve_flow_problem_highs(original_node_df, original_edge_df)    
    print(grb_metrics)
    print(highs_metrics)        
    diff = abs(highs_metrics["total_unmet_demand"] - grb_metrics["total_unmet_demand"])
    assert diff < 1e-6
    diff = abs(highs_metrics["total_unused_capacity"] - grb_metrics["total_unused_capacity"])
    assert diff < 1e-6
    # use highs 
    node_df = highs_node_df
    edge_df = highs_edge_df
    metrics = highs_metrics
    return node_df, edge_df, metrics
    # return highs_node_df, highs_edge_df, metrics

def create_canada_ui():
    st.set_page_config(page_title="Pipeline Design", layout="wide")
    st.title("Pipeline Design and Flow Optimization")

    # Input fields in the sidebar
    with st.sidebar:
        input_num_cities = st.number_input(
            "Number of cities to display (closest to Edmonton):",
            min_value=2,  # Need at least 2 cities for an edge
            max_value=100,
            value=30, 
            step=1,
            key="city_input" # Assign a key for potential future reference
        )
        average_capacity_per_city = st.number_input(
            "Average capacity per source city:",
            min_value=1,
            max_value=1000,
            value=100,
            step=1
        )
        average_demand_per_city = st.number_input(
            "Average demand per sink city:",
            min_value=1,
            max_value=1000,
            value=10,
            step=1
        )

        update_button = st.button("Run")

    config = Config(
        average_capacity_per_node=average_capacity_per_city,
        average_demand_per_node=average_demand_per_city,
        max_node_count=input_num_cities
    )
    if 'node_df' not in st.session_state or update_button:
        with st.spinner("Computing routes..."):
            st.session_state.node_df, st.session_state.edge_df, st.session_state.metrics = get_graph_data(config)


    st.subheader("Metrics")
    metrics = st.session_state.metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Unmet Demand", f"{metrics['total_unmet_demand']:.2f}")
    with col2:
        st.metric("Total Unused Capacity", f"{metrics['total_unused_capacity']:.2f}")
    with col3:
        st.metric("Is Optimal", f"{metrics['is_optimal']}")
    with col4:
        st.metric("Runtime (ms)", f"{1000 * metrics['runtime']:.2f}")

    node_df = st.session_state.node_df
    edge_df = st.session_state.edge_df
    edge_df = edge_df[edge_df['selected']].copy()
    sources_nodes = node_df[node_df["demand"] < 0]
    sinks_nodes = node_df[node_df["demand"] > 0]
    sink_nodes_demand_met = sinks_nodes[sinks_nodes['demand_met']]
    sink_nodes_demand_unmet = sinks_nodes[~sinks_nodes['demand_met']]

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
    # Trace for sources nodes
    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=sources_nodes['lat'],
        lon=sources_nodes['lon'],
        marker=dict(size=10, color="black"),
        name="Sources",
        hoverinfo='text',
        hovertext=sources_nodes.apply(lambda row: f"{row['city']}<br>Demand: {row['demand']:.2f}<br>Inbound: {row['inbound_flow']:.2f}<br>Outbound: {row['outbound_flow']:.2f}<br>Flow Difference: {row['flow_difference']:.2f}", axis=1)
    ))

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=sink_nodes_demand_met['lat'],
        lon=sink_nodes_demand_met['lon'],
        marker=dict(size=5, color="green"),
        name="Cities (Demand Met)",
        hoverinfo='text',
        hovertext=sink_nodes_demand_met.apply(lambda row: f"{row['city']}<br>Demand: {row['demand']:.2f}<br>Inbound: {row['inbound_flow']:.2f}<br>Outbound: {row['outbound_flow']:.2f}<br>Flow Difference: {row['flow_difference']:.2f}", axis=1)
    ))

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lat=sink_nodes_demand_unmet['lat'],
        lon=sink_nodes_demand_unmet['lon'],
        marker=dict(size=5, color="red"),
        name="Cities (Demand Unmet)",
        hoverinfo='text',
        hovertext=sink_nodes_demand_unmet.apply(lambda row: f"{row['city']}<br>Demand: {row['demand']:.2f}<br>Inbound: {row['inbound_flow']:.2f}<br>Outbound: {row['outbound_flow']:.2f}<br>Flow Difference: {row['flow_difference']:.2f}", axis=1)
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5,
        mapbox_center_lat = 53.5344,
        mapbox_center_lon = -113.4903,
        margin={"r":0,"t":30,"l":0,"b":0},
        height=600,
        showlegend=True
    )

    config = {'scrollZoom': True}
    st.plotly_chart(fig, use_container_width=True, config=config)

    st.subheader("Node Data")
    node_cols_to_show = [col for col in node_df.columns if 'lon' not in col.lower() and 'lat' not in col.lower()]
    st.dataframe(node_df[node_cols_to_show])

    st.subheader("Edge Data")
    edge_cols_to_show = [col for col in edge_df.columns if 'lon' not in col.lower() and 'lat' not in col.lower()]
    st.dataframe(edge_df[edge_cols_to_show])

if __name__ == "__main__":
    create_canada_ui()


