import pandas as pd 
import numpy as np

def generate_demand(node_df: pd.DataFrame, average_capacity_per_city: int = 100, average_demand_per_city: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Assigns demand to each node in the dataframe.
    Edmonton and Calgary are source nodes with negative demand (capacity).
    All other nodes are sink nodes with positive demand (need for supply).
    Demand values include some randomness for realism.
    
    Args:
        node_df (pd.DataFrame): Dataframe containing node information with 'city' column.
        average_capacity_per_city (int): Average capacity per city.
        average_demand_per_city (int): Average demand per city.
        seed (int): Seed for the random number generator.
    Returns:
        pd.DataFrame: Dataframe with added 'demand' column.
    """
    rng = np.random.default_rng(seed)

    node_df['demand'] = node_df['city'].apply(
        lambda x: -average_capacity_per_city + rng.uniform(-20, 20) if x in ['Edmonton', 'Calgary'] else average_demand_per_city + rng.uniform(-5, 5)
    )
    return node_df 
