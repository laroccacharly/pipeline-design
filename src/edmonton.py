import pandas as pd
import numpy as np
import streamlit as st

from .distance_matrix import get_distance

def compute_distance_from_edmonton(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the distance of each city from Edmonton using the precomputed distance matrix."""
    if 'id' not in df.columns:
        st.error("City data must contain an 'id' column.")
        df['distance_from_edmonton'] = np.nan
        return df

    try:
        # Find Edmonton's ID
        edmonton_id = df.loc[df['city'] == 'Edmonton', 'id'].iloc[0]
    except IndexError:
        st.error("Edmonton not found in the city data or does not have an ID. Cannot calculate distances.")
        df['distance_from_edmonton'] = np.nan
        return df

    # Apply get_distance using Edmonton's ID and each city's ID
    df['distance_from_edmonton'] = df['id'].apply(
        lambda other_id: get_distance(edmonton_id, other_id)
    )
    print(f"Looked up distances from Edmonton (ID: {edmonton_id}).")
    return df

def filter_closest_to_edmonton(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    n = min(n, len(df))
    closest_df = df.sort_values(by='distance_from_edmonton').head(n)
    print(f"Filtered to {len(closest_df)} cities closest to Edmonton.")
    return closest_df
