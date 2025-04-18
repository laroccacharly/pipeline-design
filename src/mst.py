import gurobipy as gp
from gurobipy import GRB
import itertools
import pandas as pd

def solve_mst(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Solves the Minimum Spanning Tree problem for a given set of nodes and edges.
    Updates the edge_df with a 'selected' column indicating MST edges.

    Args:
        node_df (pd.DataFrame): DataFrame with node information (must contain 'id').
        edge_df (pd.DataFrame): DataFrame with edge information ('source', 'target', 'distance').

    Returns:
        tuple: A tuple containing the original node_df and the updated edge_df
               with a 'selected' column (1 for selected, 0 otherwise).
    """
    print("Solving MST...")
    nodes = node_df['id'].tolist() # Use the 'id' column for node identifiers
    n = len(nodes)

    # Initialize 'selected' column in edge_df
    edge_df['selected'] = 0

    # Ensure edges used in the model are unique, sorted tuples based on 'source' and 'target'
    model_edges_tuples = [tuple(sorted((row['source'], row['target']))) for _, row in edge_df.iterrows()]
    model_edges_set = set(model_edges_tuples) # Use a set for efficient lookups
    model_edges_list = list(model_edges_set) # Unique edges for the model

    # Create distance dictionary using the sorted tuples as keys
    dist = {}
    for _, row in edge_df.iterrows():
        edge_tuple = tuple(sorted((row['source'], row['target'])))
        # Avoid overwriting if duplicate edges exist with different distances (use first encountered)
        if edge_tuple not in dist:
            dist[edge_tuple] = row['distance']


    try:
        # Use Kruskal's algorithm to generate the MST solution
        def find(parent, i):
            if parent[i] != i:
                parent[i] = find(parent, parent[i])
            return parent[i]

        def union(parent, rank, x, y):
            px, py = find(parent, x), find(parent, y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Initialize data structures for Kruskal's algorithm
        selected_edge_tuples = set()
        parent = {node: node for node in nodes}
        rank = {node: 0 for node in nodes}
        edges_sorted = sorted(model_edges_list, key=lambda e: dist[e])
        selected_edges = 0
        i = 0

        # Apply Kruskal's algorithm to find the MST
        while selected_edges < n - 1 and i < len(edges_sorted):
            u, v = edges_sorted[i]
            i += 1
            if find(parent, u) != find(parent, v):
                union(parent, rank, u, v)
                selected_edge_tuples.add((u, v))
                selected_edges += 1


        # Update the 'selected' column in the original edge_df
        total_distance = 0
        for index, row in edge_df.iterrows():
            edge_tuple = tuple(sorted((row['source'], row['target'])))
            if edge_tuple in selected_edge_tuples:
                edge_df.loc[index, 'selected'] = 1
                total_distance += row['distance']
        # Convert 'selected' column to boolean type
        edge_df['selected'] = edge_df['selected'].astype(bool)
        
        return node_df, edge_df

    except Exception as e:
        print(f'Error during MST computation with Kruskal\'s algorithm: {e}')
        raise  # Re-raise the exception