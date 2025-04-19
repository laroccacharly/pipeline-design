import highspy
import pandas as pd
import networkx as nx
from .create_edge_df import duplicate_edges

def solve_flow_problem_highs(node_df: pd.DataFrame, edge_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, any]]:
    """
    Solve the flow problem using HiGHS high-level API.

    Args:
        node_df: DataFrame with node information (id, demand).
        edge_df: DataFrame with undirected edge information (source, target, distance).

    Returns:
        A tuple containing:
        - Updated node_df with flow information.
        - Updated edge_df with flow information (only edges with flow > 0).
        - A dictionary with optimization metrics.
    """
    # Create a directed graph
    edge_df_directed = duplicate_edges(edge_df.copy())
    G = nx.from_pandas_edgelist(edge_df_directed, source='source', target='target', edge_attr=True, create_using=nx.DiGraph())
    
    # Add node attributes
    for _, row in node_df.iterrows():
        G.add_node(row['id'], demand=row['demand'])

    node_list = list(G.nodes())
    edge_list = list(G.edges())

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

    h = highspy.Highs()
    inf = highspy.kHighsInf

    # --- Define Variables using high-level API ---
    flow_vars = {}
    for edge in edge_list:
        # Assuming a small cost for flow, similar to original
        flow_vars[edge] = h.addVariable(lb=0, ub=inf) 

    unmet_demand_vars = {}
    for node_id in sink_node_ids:
        unmet_demand_vars[node_id] = h.addVariable(lb=0, ub=inf)

    unused_capacity_vars = {}
    for node_id in source_node_ids:
        unused_capacity_vars[node_id] = h.addVariable(lb=0, ub=inf)

    # --- Define Objective Function using high-level API ---
    flow_cost = 0.1 # Cost per unit of flow
    unmet_cost = 1.0 # Cost per unit of unmet demand
    # unused_cost = 0.0 # Cost per unit of unused capacity (implicitly zero if not added to objective)

    objective = h.qsum(flow_vars[edge] * flow_cost for edge in edge_list) + \
                h.qsum(unmet_demand_vars[node_id] * unmet_cost for node_id in sink_node_ids)
                # No cost for unused capacity, so we don't add unused_capacity_vars here
                # + h.qsum(unused_capacity_vars[node_id] * unused_cost for node_id in source_node_ids)

    h.minimize(objective)

    # --- Define Constraints using high-level API ---

    # Conservation constraints for source nodes
    for node_id in source_node_ids:
        node_capacity = -G.nodes[node_id]['demand'] # Demand is negative, make it positive capacity
        outgoing_flow = h.qsum(flow_vars[edge] for edge in G.out_edges(node_id))
        h.addConstr(outgoing_flow + unused_capacity_vars[node_id] == node_capacity)

    # Conservation constraints for sink nodes
    for node_id in sink_node_ids:
        node_demand = G.nodes[node_id]['demand'] # Demand is positive
        ingoing_flow = h.qsum(flow_vars[edge] for edge in G.in_edges(node_id))
        outgoing_flow = h.qsum(flow_vars[edge] for edge in G.out_edges(node_id)) # Flow can leave sink nodes
        h.addConstr(ingoing_flow - outgoing_flow + unmet_demand_vars[node_id] == node_demand)

    # Conditional capacity/demand balancing constraints
    if total_demand > total_capacity:
        # Force total_unused_capacity == 0
        total_unused = h.qsum(unused_capacity_vars[node_id] for node_id in source_node_ids)
        h.addConstr(total_unused == 0)
        print("Added constraint: force_capacity_use")

    if total_demand < total_capacity:
        # Force total_unmet_demand == 0
        total_unmet = h.qsum(unmet_demand_vars[node_id] for node_id in sink_node_ids)
        h.addConstr(total_unmet == 0)
        print("Added constraint: force_demand_use")

    # --- Solve --- 
    # h.setOptionValue('time_limit', 60.0) # Set time limit if needed
    h.run()
    _ = h.getInfo()
    model_status = h.getModelStatus()
    runtime = h.getRunTime()
    print(f"Optimization completed in {runtime:.2f} seconds.")
    print("Status: ", h.modelStatusToString(model_status))
    is_optimal = (model_status == highspy.HighsModelStatus.kOptimal)
    print("Is optimal: ", is_optimal)

    # --- Process Results ---
    acceptable_statuses = [
        highspy.HighsModelStatus.kOptimal,
        highspy.HighsModelStatus.kObjectiveBound,
        highspy.HighsModelStatus.kObjectiveTarget
    ]
    if model_status not in acceptable_statuses:
        status_str = h.modelStatusToString(model_status)
        raise RuntimeError(f"Optimization failed or did not find an acceptable solution. Status: {status_str}")

    # Get solution values using h.vals()
    flow_values = h.vals(flow_vars)
    unmet_demand_values = h.vals(unmet_demand_vars)
    unused_capacity_values = h.vals(unused_capacity_vars)

    # Set flow values on directed edge dataframe
    edge_df_directed['flow'] = 0.0
    for edge, flow_val in flow_values.items():
        edge_df_directed.loc[(edge_df_directed['source'] == edge[0]) & (edge_df_directed['target'] == edge[1]), 'flow'] = flow_val

    total_unmet_demand = sum(unmet_demand_values.values())
    print(f"Total unmet demand: {total_unmet_demand:.2f}")

    total_unused_capacity = sum(unused_capacity_values.values())
    print(f"Total unused capacity: {total_unused_capacity:.2f}")

    node_df_indexed = node_df.set_index('id')
    node_df_indexed['inbound_flow'] = 0.0
    node_df_indexed['outbound_flow'] = 0.0

    # Aggregate flow for each node using the index
    for (source, target), flow in flow_values.items():
        if source in node_df_indexed.index:
                node_df_indexed.loc[source, 'outbound_flow'] += flow
        if target in node_df_indexed.index:
                node_df_indexed.loc[target, 'inbound_flow'] += flow

    node_df = node_df_indexed.reset_index()

    node_df['flow_difference'] = node_df['inbound_flow'] - node_df['outbound_flow']
    # Adjust tolerance for floating point comparison
    tolerance = 1e-6 
    # Ensure demand is numeric for comparison
    node_df['demand'] = pd.to_numeric(node_df['demand'], errors='coerce') 
    node_df['demand_met'] = node_df.apply(
        lambda row: abs(row['flow_difference'] - row['demand']) < tolerance if pd.notna(row['demand']) else False, 
        axis=1
    )
    
    # Filter out edges where flow is near zero for the final output
    edge_df_filtered = edge_df_directed[edge_df_directed['flow'] > tolerance].copy()
    print("Edges count with flow > 0: ", len(edge_df_filtered))
    print(edge_df_filtered.head()) 

    metrics = {
        'total_unmet_demand': total_unmet_demand,
        'total_unused_capacity': total_unused_capacity,
        'is_optimal': is_optimal,
        'runtime': runtime,
        'objective_value': h.getObjectiveValue()
    }

    return node_df, edge_df_filtered, metrics