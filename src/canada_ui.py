from .data import load_canada_cities_df
import streamlit as st

def create_canada_ui():
    df = load_canada_cities_df()

    # Column renaming now happens in src/data.py

    st.title("Canadian Cities Map")
    st.map(df)


if __name__ == "__main__":
    create_canada_ui()


