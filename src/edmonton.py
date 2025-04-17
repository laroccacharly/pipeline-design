import pandas as pd
from .distance_matrix import get_distance

def compute_distance_from_edmonton(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the distance of each city from Edmonton using the precomputed distance matrix."""
    edmonton_id = df.loc[df['city'] == 'Edmonton', 'id'].iloc[0]
    # Apply get_distance using Edmonton's ID and each city's ID
    df['distance_from_edmonton'] = df['id'].apply(
        lambda other_id: get_distance(edmonton_id, other_id)
    )
    return df

def filter_closest_to_edmonton(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    n = min(n, len(df))
    closest_df = df.sort_values(by='distance_from_edmonton').head(n)
    print(f"Filtered to {len(closest_df)} cities closest to Edmonton.")
    return closest_df
