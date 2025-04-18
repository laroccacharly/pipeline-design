import pandas as pd 
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

def duplicate_edges(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
        Duplicate the edges to make it a directed graph.
    """
    reverse_source_target = edge_df.copy()
    reverse_source_target['source'] = edge_df['target']
    reverse_source_target['target'] = edge_df['source']
    return pd.concat([edge_df, reverse_source_target], ignore_index=True)

def solve_flow_problem(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        edge_df is a df of undirected edges. We need to conver to networkx and then add the reverse edge for each edge to make it a directed graph.
        No capacity on edges. 
        Solve the flow problem using Gurobi. 
    """
    # Create a directed graph instead of undirected
    edge_df = duplicate_edges(edge_df)
    # Add edges in both directions
    G = nx.from_pandas_edgelist(edge_df, source='source', target='target', edge_attr=True, create_using=nx.DiGraph())
    # Add node attributes
    for _, row in node_df.iterrows():
        G.add_node(row['id'], demand=row['demand'])

    sink_node_ids = [node for node in G.nodes() if G.nodes[node]['demand'] >= 0]
    source_node_ids = [node for node in G.nodes() if G.nodes[node]['demand'] < 0]
    total_demand = sum(G.nodes[node]['demand'] for node in sink_node_ids)
    total_capacity = -sum(G.nodes[node]['demand'] for node in source_node_ids)
    print("Total demand: ", total_demand)
    print("Total capacity: ", total_capacity)
    if total_demand > total_capacity:
        print("Bounded by capacity")
    else:
        print("Bounded by demand")

    model = gp.Model('FlowProblem')
    model.setParam('TimeLimit', 60)

    # Variables 

    # Flow 
    flow = model.addVars(G.edges(), lb=0, name='flow')
    # Unmet demand
    unmet_demand = model.addVars(sink_node_ids, lb=0, name='unmet_demand')
    # Unused capacity
    unused_capacity = model.addVars(source_node_ids, lb=0, name='unused_capacity')
    total_flow = gp.quicksum(flow[edge] for edge in G.edges())
    total_unmet_demand = gp.quicksum(unmet_demand[node] for node in sink_node_ids)
    total_unused_capacity = gp.quicksum(unused_capacity[node] for node in source_node_ids)
    # total_cost = gp.quicksum(G.edges[edge]['distance'] * flow[edge] for edge in G.edges())
    total_cost = total_unmet_demand + total_flow * 0.1
    model.setObjective(total_cost, GRB.MINIMIZE)
    # Force total capacity use when bounded by capacity
    if total_demand > total_capacity:
        model.addConstr(total_unused_capacity == 0, name="force_capacity_use")

    if total_demand < total_capacity:
        model.addConstr(total_unmet_demand == 0, name="force_demand_use")


    # Conversation on source nodes
    for node in source_node_ids:
        outflow_edges = G.out_edges(node)
        outflow_flow = gp.quicksum(flow[edge] for edge in outflow_edges)
        model.addConstr(outflow_flow + unused_capacity[node] == -G.nodes[node]['demand'])

    # Conversation on sink nodes
    for node in sink_node_ids:
        inflow_edges = G.in_edges(node)
        outflow_edges = G.out_edges(node)
        inflow_flow = gp.quicksum(flow[edge] for edge in inflow_edges)
        outflow_flow = gp.quicksum(flow[edge] for edge in outflow_edges)
        model.addConstr(inflow_flow + unmet_demand[node] == outflow_flow + G.nodes[node]['demand'])

    # Optimize
    model.optimize()

    # Show status 
    print(f"Optimization completed in {model.Runtime:.2f} seconds.")
    print("Status: ", model.status)
    is_optimal = model.status == GRB.OPTIMAL
    print("Is optimal: ", is_optimal)

    # set flow values 
    for id, row in edge_df.iterrows():
        edge_df.at[id, 'flow'] = flow[row['source'], row['target']].x

    # Total unmet demand
    total_unmet_demand = sum(unmet_demand[node].x for node in sink_node_ids)
    print(f"Total unmet demand: {total_unmet_demand:.2f}")

    # Total unused capacity
    total_unused_capacity = sum(unused_capacity[node].x for node in source_node_ids)
    print(f"Total unused capacity: {total_unused_capacity:.2f}")

    # Calculate inbound and outbound flow for each node
    for node_id in node_df["id"]: 
        inbound_edges = G.in_edges(node_id)
        outbound_edges = G.out_edges(node_id)
        node_df.at[node_id, 'inbound_flow'] = sum(flow[edge].x for edge in inbound_edges)
        node_df.at[node_id, 'outbound_flow'] = sum(flow[edge].x for edge in outbound_edges)
    
    node_df['flow_difference'] = node_df['inbound_flow'] - node_df['outbound_flow']
    node_df['demand_met'] = node_df.apply(lambda row: abs(row['flow_difference'] - row['demand']) < 0.01, axis=1)

    # all sink nodes where outflow is greater than inflow 
    sink_nodes = node_df[node_df['demand'] > 0]
    sink_nodes_with_excess_outflow = sink_nodes[sink_nodes['outbound_flow'] > sink_nodes['inbound_flow']]
    print(f"Sink nodes with excess outflow: {len(sink_nodes_with_excess_outflow)}")
    print(sink_nodes_with_excess_outflow)

    # all sources nodes
    source_nodes = node_df[node_df['demand'] < 0]
    print("Source nodes:")
    print(source_nodes)

    # Filter out edges where flow is 0
    edge_df = edge_df[edge_df['flow'] > 0]
    print("Edges count with flow > 0: ", len(edge_df))  
    print(edge_df)
    return node_df, edge_df


    