import os
import requests
import zipfile
import pandas as pd

data_dir = "data"
zip_url = "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.77.zip"
csv_filename = "worldcities.csv"
csv_path = os.path.join(data_dir, csv_filename)
zip_path = os.path.join(data_dir, "worldcities.zip")
canada_parquet_filename = "canada_cities.parquet"
canada_parquet_path = os.path.join(data_dir, canada_parquet_filename)

def download_world_cities_data():
    """Downloads and extracts the world cities CSV if it doesn't exist."""
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download and unzip if CSV doesn't exist
    if not os.path.exists(csv_path):
        print(f"Downloading data from {zip_url}...")
        response = requests.get(zip_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded to {zip_path}")

        print(f"Unzipping {zip_path} to {data_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only the specific CSV file we need
            zip_ref.extract(csv_filename, data_dir)
            # Optionally, extract other files like the license if needed
            # zip_ref.extractall(data_dir)
        print(f"Unzipped {csv_filename}")

        # Clean up the zip file
        os.remove(zip_path)
        print(f"Removed {zip_path}")
    else:
        print(f"{csv_path} already exists. Skipping download.")


def load_world_cities_df() -> pd.DataFrame:
    """Loads the world cities data from the CSV file."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}.")
    return df


def filter_and_select_canada_cities(df: pd.DataFrame) -> pd.DataFrame:
    """Filters the DataFrame for Canadian cities and selects relevant columns."""
    print("Filtering for Canadian cities...")
    canada_df = df[df['country'] == 'Canada'].copy()

    # Select longitude and latitude
    lon_lat_df = canada_df[['city', 'lng', 'lat']].reset_index(drop=True)

    # Rename 'lng' to 'lon' for consistency (e.g., with Streamlit)
    lon_lat_df = lon_lat_df.rename(columns={'lng': 'lon'})

    print(f"Filtered data to {len(lon_lat_df)} Canadian cities.")
    return lon_lat_df


# main function
def load_canada_cities_df() -> pd.DataFrame:
    """Loads Canadian city data, using a parquet cache if available."""
    # Check if the cached parquet file exists
    if os.path.exists(canada_parquet_path):
        print(f"Loading cached data from {canada_parquet_path}...")
        canada_cities_df = pd.read_parquet(canada_parquet_path)
        print(f"Loaded {len(canada_cities_df)} records from cache.")
        return canada_cities_df

    # If cache doesn't exist, proceed with download, load, filter
    print(f"Cache not found at {canada_parquet_path}. Processing data...")
    download_world_cities_data()
    world_cities_df = load_world_cities_df()
    canada_cities_df = filter_and_select_canada_cities(world_cities_df)

    # Save the filtered data to parquet cache
    print(f"Saving filtered data to {canada_parquet_path}...")
    canada_cities_df.to_parquet(canada_parquet_path, index=False)
    print("Saved cache.")

    return canada_cities_df

if __name__ == '__main__':
    # Example usage:
    canadian_city_coords = load_canada_cities_df() 
    print("\nFirst 5 Canadian cities:")
    print(canadian_city_coords.head())
    print("\nLast 5 Canadian cities:")
    print(canadian_city_coords.tail()) 