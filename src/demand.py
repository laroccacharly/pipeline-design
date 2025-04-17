import pandas as pd 
import numpy as np

# Set a global seed for reproducibility
np.random.seed(42)

def generate_demand(node_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns demand to each node in the dataframe.
    Edmonton and Calgary are source nodes with negative demand (capacity).
    All other nodes are sink nodes with positive demand (need for supply).
    Demand values include some randomness for realism.
    
    Args:
        node_df (pd.DataFrame): Dataframe containing node information with 'city' column.
    
    Returns:
        pd.DataFrame: Dataframe with added 'demand' column.
    """
    node_df['demand'] = node_df['city'].apply(
        lambda x: -100 + np.random.uniform(-20, 20) if x in ['Edmonton', 'Calgary'] else 10 + np.random.uniform(-5, 5)
    )
    return node_df 
