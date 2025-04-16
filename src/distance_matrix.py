import pandas as pd
import numpy as np
import os
from .haversine import haversine

# Define path globally
_DATA_DIR = "data"
_PARQUET_PATH = os.path.join(_DATA_DIR, "distance_matrix.parquet")

# In-memory cache for O(1) lookups once loaded
_distance_matrix_hash = {}

def create_distance_matrix(node_df: pd.DataFrame, output_path: str = _PARQUET_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ids = node_df['id'].values
    lats = node_df['lat'].values
    lons = node_df['lon'].values

    # Get indices for the upper triangle of the matrix (k=1 excludes the diagonal)
    indices = np.triu_indices(len(node_df), k=1)
    city_A_indices = indices[0]
    city_B_indices = indices[1]

    # Select the corresponding IDs and coordinates using the indices
    ids_A = ids[city_A_indices]
    ids_B = ids[city_B_indices]
    lats_A = lats[city_A_indices]
    lons_A = lons[city_A_indices]
    lats_B = lats[city_B_indices]
    lons_B = lons[city_B_indices]

    # Perform vectorized distance calculation
    distances = haversine(lats_A, lons_A, lats_B, lons_B)

    # Create the long-format DataFrame
    dist_df = pd.DataFrame({
        'city_A_id': ids_A,
        'city_B_id': ids_B,
        'distance': distances
    })

    print(f"Saving {len(dist_df)} distance pairs to {output_path}...")
    dist_df.to_parquet(output_path, index=False, compression='snappy') # Snappy is fast
    print("Distance matrix saved.")
    # Clear the cache as the underlying data source has been updated
    _distance_matrix_hash.clear()

def load_distance_matrix(input_path: str = _PARQUET_PATH) -> pd.DataFrame:
    """
    Loads the distance matrix from the parquet file.

    Args:
        input_path: Path to the Parquet file. Defaults to data/distance_matrix.parquet.

    Returns:
        DataFrame containing the distance matrix in long format.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    print(f"Loading distance matrix from {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Distance matrix file not found at {input_path}. "
                              "Consider running create_distance_matrix first.")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} distance pairs.")
    return df

def _populate_distance_hash_from_file(input_path: str = _PARQUET_PATH):
    """Loads data and populates the internal hash cache. Internal use only."""
    global _distance_matrix_hash
    # Only load and populate if the cache is currently empty
    if not _distance_matrix_hash:
        print("Populating distance hash cache from file...")
        try:
            dist_df = load_distance_matrix(input_path)
            # Iterate through the DataFrame rows to build the hash map
            for _, row in dist_df.iterrows():
                # Use tuple of (min_id, max_id) as key for symmetry
                key = (min(int(row['city_A_id']), int(row['city_B_id'])),
                       max(int(row['city_A_id']), int(row['city_B_id'])))
                _distance_matrix_hash[key] = row['distance']
            print(f"Distance hash cache populated with {len(_distance_matrix_hash)} entries.")
        except FileNotFoundError as e:
            # If the file doesn't exist, we can't populate the cache.
            print(f"Warning: Could not populate hash cache. {e}")
            # Ensure cache remains empty
            _distance_matrix_hash.clear()
        except Exception as e:
             print(f"Error populating distance hash cache: {e}")
             _distance_matrix_hash.clear()


def get_distance(node_id: int, other_node_id: int) -> float:
    """
    Uses the hash cache to get the distance between two nodes (O(1) average time complexity).
    Populates the cache from the file automatically if it's empty.

    Args:
        node_id: ID of the first node.
        other_node_id: ID of the second node.

    Returns:
        The distance in kilometers,
        0.0 if node_id equals other_node_id,
        float('inf') if the distance pair is not found in the cache (e.g., file missing or invalid IDs).
    """
    node_id = int(node_id)
    other_node_id = int(other_node_id)

    if node_id == other_node_id:
        return 0.0

    # Ensure cache is populated if it's currently empty
    # This makes the first call potentially slow, but subsequent calls fast.
    if not _distance_matrix_hash:
         _populate_distance_hash_from_file() # Attempt to load from default path

    # Create the lookup key, ensuring order (min_id, max_id)
    key = (min(node_id, other_node_id), max(node_id, other_node_id))

    # Lookup in the hash, return infinity if key is not found
    distance = _distance_matrix_hash.get(key, float('inf'))

    # Optional: Add a warning if distance is infinity to help debugging
    # if distance == float('inf'):
    #    print(f"Warning: Distance between {node_id} and {other_node_id} not found in cache. Key: {key}")

    return distance

