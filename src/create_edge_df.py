import pandas as pd 
import itertools
from .distance_matrix import get_distance

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

def duplicate_edges(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
        Duplicate the edges to make it a directed graph.
    """
    reverse_source_target = edge_df.copy()
    reverse_source_target['source'] = edge_df['target']
    reverse_source_target['target'] = edge_df['source']
    return pd.concat([edge_df, reverse_source_target], ignore_index=True)