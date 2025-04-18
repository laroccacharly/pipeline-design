import pandas as pd 
import gurobipy as gp
from gurobipy import GRB
import time

def solve_flow_problem(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Solves the flow problem using Gurobi.
    Constraints: Inbound flow = Outbound flow + demand for each node, with slack for unmet demand or excess capacity.
    Objective: Minimize total unmet demand.
    Edges don't have capacity (infinite).
    
    Args:
        node_df (pd.DataFrame): DataFrame with node information including 'id' and 'demand'.
        edge_df (pd.DataFrame): DataFrame with edge information including 'source', 'target', and 'distance'.
    
    Returns:
        tuple: (node_df, edge_df) with edge_df updated to include a 'flow' column.
    """
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
    
    # Add flow variables for each edge (non-negative flow since edges have infinite capacity)
    flow = model.addVars(edges, lb=0, name='flow')
    
    # Add slack variables for each node to account for unmet demand (positive) or excess capacity (negative)
    slack = model.addVars(nodes, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='slack')
    
    # Flow conservation constraint: Inbound flow - Outbound flow + slack = demand for each node
    for node in nodes:
        inbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if j == node)
        outbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if i == node)
        model.addConstr(inbound_flow - outbound_flow + slack[node] == demand[node], name=f'flow_conservation_{node}')
    
    # Objective: Minimize absolute slack on sink nodes (nodes with positive demand)
    # Create auxiliary variables for absolute value of slack for sink nodes only
    sink_nodes = [node for node in nodes if demand[node] > 0]
    abs_slack = model.addVars(sink_nodes, lb=0, name='abs_slack')
    for node in sink_nodes:
        model.addConstr(abs_slack[node] >= slack[node], name=f'abs_slack_pos_{node}')
        model.addConstr(abs_slack[node] >= -slack[node], name=f'abs_slack_neg_{node}')

    # Total outflow must be smaller than inflow for each sink nodes (cannot produce more) 
    #for node in sink_nodes:
    #    inbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if j == node)
    #    outbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if i == node)
    #    model.addConstr(outbound_flow + demand[node] <= inbound_flow, name=f'inflow_outflow_balance_{node}')

    source_nodes = [node for node in nodes if demand[node] < 0]
    # add constract such that total outflow from source nodes is bounded by the capacity (negative demand) of the node
    for node in source_nodes:
        outbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if i == node)
        model.addConstr(outbound_flow <= -demand[node], name=f'source_node_outflow_bound_{node}')


    model.setObjective(
        gp.quicksum(abs_slack[node] for node in sink_nodes), 
        GRB.MINIMIZE
    )
    
    # Optimize the model
    model.optimize()
    
    print(f"Optimization completed in {model.Runtime:.2f} seconds.")
    print("Status: ", model.status)

    if model.solCount == 0:
        print("Model is infeasible. No solution exists under current constraints.")
        model.computeIIS()
        print("Infeasible constraints:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(f" - {c.ConstrName}")

        raise Exception("Model is infeasible. No solution exists under current constraints.")
    
    # Update edge_df with flow values if solution exists

    edge_df['flow'] = 0.0
    for edge in edges:
        idx = edge_indices[edge]
        edge_df.at[idx, 'flow'] = flow[edge].x
    print(f"Updated edge_df with flow values.")
    # Optionally log total unmet demand
    total_slack_on_sink_nodes = sum(abs(slack[node].x) for node in sink_nodes)
    total_slack_on_source_nodes = sum(abs(slack[node].x) for node in source_nodes)
    print(f"Total slack on sink nodes: {total_slack_on_sink_nodes:.2f}")
    print(f"Total slack on source nodes: {total_slack_on_source_nodes:.2f}")
    
    
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
    node_df['flow_difference'] = node_df['inbound_flow'] - node_df['outbound_flow']
    node_df['demand_met'] = node_df.apply(lambda row: abs(row['flow_difference'] - row['demand']) < 0.01, axis=1)
    # all sink nodes where outflow is greater than inflow 
    sink_nodes = node_df[node_df['demand'] > 0]
    sink_nodes_with_excess_outflow = sink_nodes[sink_nodes['outbound_flow'] > sink_nodes['inbound_flow']]
    print(f"Sink nodes with excess outflow: {len(sink_nodes_with_excess_outflow)}")
    print(sink_nodes_with_excess_outflow)
    
    
    return node_df, edge_df

