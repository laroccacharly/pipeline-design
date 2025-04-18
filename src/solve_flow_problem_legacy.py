import pandas as pd 
import gurobipy as gp
from gurobipy import GRB
import time
import networkx as nx

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
    # Convert node_df and edge_df to networkx graph
    G = nx.from_pandas_edgelist(edge_df, source='source', target='target')
    # Add node demand as a node attribute
    for idx, row in node_df.iterrows():
        G.add_node(row['id'], demand=row['demand'])
    
    # Create a flow problem using networkx
    flow_problem = nx.max_weight_matching(G, maxcardinality=True)
    
    


    # Initialize the Gurobi model
    model = gp.Model('FlowProblem')
    
    # Set a time limit of 60 seconds
    model.setParam('TimeLimit', 60)
    
    # Create node id list
    nodes_ids = node_df['id'].tolist()
    
    # Create dictionaries for demand
    demand = dict(zip(node_df['id'], node_df['demand']))
    
    # Create edge list as tuples for flow variables
    # Exclude edges originating from sink nodes
    sink_node_ids = [node for node in nodes_ids if demand[node] >= 0]
    source_node_ids = [node for node in nodes_ids if demand[node] < 0]

    edges: list[tuple[str, str]] = [] 
    edge_indices: dict[tuple[str, str], int] = {}
    for id, row in edge_df.iterrows():
        edge = (row['source'], row['target'])
        edge_reverse = (row['target'], row['source'])
        edges.append(edge)
        edges.append(edge_reverse)
        edge_indices[edge] = id
        edge_indices[edge_reverse] = id
    
    # Add flow variables for each edge
    flow = model.addVars(edges, lb=0, name='flow')
    
    unmet_demand = model.addVars(sink_nodes, lb=0, name='unmet_demand')
    unused_capacity = model.addVars(source_nodes, lb=0, name='unused_capacity')
    # Flow conservation constraint: Inbound flow - Outbound flow + slack = demand for each node
    for node in sink_nodes:
        inbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if j == node)
        outbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if i == node)
        model.addConstr(inbound_flow - outbound_flow + slack[node] == demand[node], name=f'flow_conservation_{node}')
    
    # Objective: Minimize unmet demand at sinks and unused capacity at sources.
    # Auxiliary variable for unmet demand (max(0, slack_sink))
    unmet_demand_slack = model.addVars(sink_nodes, lb=0, name='unmet_demand_slack')
    for node in sink_nodes:
        model.addConstr(unmet_demand_slack[node] >= slack[node], name=f'unmet_demand_pos_{node}')
        # No constraint for unmet_demand_slack >= -slack needed, as we only penalize positive slack

    # Auxiliary variable for unused capacity (max(0, -slack_source))
    unused_capacity_slack = model.addVars(source_nodes, lb=0, name='unused_capacity_slack')
    for node in source_nodes:
        # We want to penalize slack < 0, which is equivalent to penalizing -slack > 0
        model.addConstr(unused_capacity_slack[node] >= -slack[node], name=f'unused_capacity_pos_{node}')
        # No constraint for unused_capacity_slack >= slack needed

    # add constract such that total outflow from source nodes is bounded by the capacity (negative demand) of the node
    for node in source_nodes:
        outbound_flow = gp.quicksum(flow[(i, j)] for (i, j) in edges if i == node)
        model.addConstr(outbound_flow <= -demand[node], name=f'source_node_outflow_bound_{node}')

    model.setObjective(
        gp.quicksum(unmet_demand_slack[node] for node in sink_nodes) + 
        gp.quicksum(unused_capacity_slack[node] for node in source_nodes), 
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
    # sink nodes with excess inflow, where "flow_difference" is larger than demand
    sink_nodes_with_excess_inflow = sink_nodes[sink_nodes['flow_difference'] > sink_nodes['demand']]
    print(f"Sink nodes with excess inflow: {len(sink_nodes_with_excess_inflow)}")
    print(sink_nodes_with_excess_inflow)
    # all sources nodes
    source_nodes = node_df[node_df['demand'] < 0]
    print(source_nodes)
    
    return node_df, edge_df

