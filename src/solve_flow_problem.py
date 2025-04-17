import pandas as pd 
import gurobipy as gp
from gurobipy import GRB
import time

def solve_flow_problem(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Solves the flow problem using Gurobi.
    Constraints: Inbound flow = Outbound flow + demand for each node, with slack for unmet demand or excess capacity.
    Objective: Minimize the total cost of the flow plus a penalty for unmet demand.
    Cost is the distance between the nodes times the flow on the edge.
    Respect demand values on the nodes (negative for source, positive for sink).
    Edges don't have capacity (infinite).
    
    Args:
        node_df (pd.DataFrame): DataFrame with node information including 'id' and 'demand'.
        edge_df (pd.DataFrame): DataFrame with edge information including 'source', 'target', and 'distance'.
    
    Returns:
        tuple: (node_df, edge_df) with edge_df updated to include a 'flow' column.
    """
    start_time = time.time()
    print("Starting flow problem optimization...")
    
    # Initialize the Gurobi model
    model = gp.Model('FlowProblem')
    
    # Set a time limit of 60 seconds
    model.setParam('TimeLimit', 60)
    
    # Create node id list
    nodes = node_df['id'].tolist()
    
    # Create dictionaries for demand
    demand = dict(zip(node_df['id'], node_df['demand']))
    
    # Create edge list as tuples for flow variables
    edges = [(row['source'], row['target']) for _, row in edge_df.iterrows()]
    edge_indices = dict(((row['source'], row['target']), idx) for idx, row in edge_df.iterrows())
    
    # Create cost dictionary (distance for each edge)
    cost = dict(((row['source'], row['target']), row['distance']) for _, row in edge_df.iterrows())
    
    # Add flow variables for each edge (non-negative flow since edges have infinite capacity)
    flow = model.addVars(edges, lb=0, name='flow')
    
    # Add slack variables for each node to account for unmet demand (positive) or excess capacity (negative)
    slack = model.addVars(nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='slack')
    
    # Flow conservation constraint: Inbound flow - Outbound flow + slack = demand for each node
    for node in nodes:
        inbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if j == node)
        outbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if i == node)
        model.addConstr(inbound_flow - outbound_flow + slack[node] == demand[node], name=f'flow_conservation_{node}')
    
    # Objective: Minimize total cost (flow * distance) plus a large penalty for unmet demand (absolute slack)
    penalty = 1000  # Large penalty for unmet demand or excess capacity
    # Create auxiliary variables for absolute value of slack
    abs_slack = model.addVars(nodes, lb=0, name='abs_slack')
    for node in nodes:
        model.addConstr(abs_slack[node] >= slack[node], name=f'abs_slack_pos_{node}')
        model.addConstr(abs_slack[node] >= -slack[node], name=f'abs_slack_neg_{node}')
    model.setObjective(
        gp.quicksum(flow[edge] * cost[edge] for edge in edges) + 
        penalty * gp.quicksum(abs_slack[node] for node in nodes), 
        GRB.MINIMIZE
    )
    
    # Optimize the model
    model.optimize()
    
    # Log results
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Optimization completed in {total_time:.2f} seconds.")
    if model.status == GRB.OPTIMAL:
        print(f"Optimal solution found with objective value: {model.objVal:.2f}")
    elif model.status == GRB.TIME_LIMIT:
        print(f"Time limit reached. Best objective value found: {model.objVal if model.solCount > 0 else 'N/A'}")
    else:
        print(f"Optimization terminated with status {model.status}. Objective value: {model.objVal if model.solCount > 0 else 'N/A'}")
    
    # Update edge_df with flow values if solution exists
    if model.solCount > 0:
        edge_df['flow'] = 0.0
        for edge in edges:
            idx = edge_indices[edge]
            edge_df.at[idx, 'flow'] = flow[edge].x
        print(f"Updated edge_df with flow values.")
        # Optionally log total unmet demand
        total_unmet = sum(abs(slack[node].x) for node in nodes)
        print(f"Total unmet demand or excess capacity: {total_unmet:.2f}")
        
        # Calculate inbound and outbound flow for each node
        node_df['inbound_flow'] = 0.0
        node_df['outbound_flow'] = 0.0
        node_df.set_index('id', inplace=True)
        for _, edge in edge_df.iterrows():
            source_id = edge['source']
            target_id = edge['target']
            flow_val = edge['flow']
            if source_id in node_df.index:
                node_df.at[source_id, 'outbound_flow'] += flow_val
            if target_id in node_df.index:
                node_df.at[target_id, 'inbound_flow'] += flow_val
        node_df.reset_index(inplace=True)
        
        # Calculate actual demand served (inbound - outbound) and whether it matches requested demand
        node_df['actual_demand_served'] = node_df['inbound_flow'] - node_df['outbound_flow']
        node_df['demand_met'] = node_df.apply(lambda row: abs(row['actual_demand_served'] - row['demand']) < 0.01, axis=1)
        print(f"Updated node_df with flow calculations and demand matching status.")
    else:
        print("No solution found. Flow values not updated.")
        edge_df['flow'] = 0.0
    
    return node_df, edge_df

