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
        # Create a new model
        m = gp.Model("mst")
        m.setParam('TimeLimit', 10)
        m.setParam('MIPFocus', 1)

        # Create variables: x[i, j] = 1 if edge (i, j) is selected, 0 otherwise
        # Variables are created based on the unique, sorted edge tuples
        x = m.addVars(model_edges_list, vtype=GRB.BINARY, name="x")

        # Set objective: minimize total distance using the unique edges
        m.setObjective(gp.quicksum(dist[i, j] * x[i, j] for i, j in model_edges_list), GRB.MINIMIZE)

        # Constraint: Exactly n-1 edges must be selected
        m.addConstr(x.sum() == n - 1, "NumEdges")

        # Constraints: Subtour elimination (Cut-set formulation)
        for k in range(2, n): # Subsets of size 2 to n-1
            for subset in itertools.combinations(nodes, k):
                # Edges within the subset, considering only existing unique edges
                subset_model_edges = [
                    tuple(sorted(edge)) for edge in itertools.combinations(subset, 2)
                    if tuple(sorted(edge)) in model_edges_set
                ]
                if subset_model_edges: # Only add constraint if there are edges within the subset
                    m.addConstr(gp.quicksum(x[i, j] for i, j in subset_model_edges) <= k - 1, f"Subtour_{k}_{'_'.join(map(str, subset))}")

        m.optimize()

        # Check if any solution was found (optimal or suboptimal)
        print(f"Solution count: {m.SolCount}")
        if m.SolCount > 0:
            print(f"MST Optimization Successful (Status: {m.Status}).") # Indicate status
            print(f"Optimization Runtime: {m.Runtime:.4f} seconds")
            print(f"Total MST Distance: {m.ObjVal:.2f}")
            selected_edge_tuples = {edge for edge in model_edges_list if x[edge].X > 0.5}

            # Update the 'selected' column in the original edge_df
            for index, row in edge_df.iterrows():
                edge_tuple = tuple(sorted((row['source'], row['target'])))
                if edge_tuple in selected_edge_tuples:
                    edge_df.loc[index, 'selected'] = 1
            # Convert 'selected' column to boolean type
            edge_df['selected'] = edge_df['selected'].astype(bool)
            m.close()
            return node_df, edge_df
        else:
            # Raise an exception if no solution was found
            raise RuntimeError(f"MST Optimization failed to find a solution (Status: {m.Status})")

    except gp.GurobiError as e:
        print(f'Gurobi error during MST optimization: {e}')
        raise # Re-raise the GurobiError
    except AttributeError as e:
        print(f'Attribute error during MST optimization: {e}')
        raise # Re-raise the AttributeError