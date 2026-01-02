import pandas as pd
import streamlit as st


@st.cache_data
def load_data(file_path="data/main-dataset.csv"):
    """
    Loads the main dataset from a CSV file, drops unnamed columns, and returns a DataFrame.
    """
    df = pd.read_csv(file_path)
    # Drop unnamed columns that are sometimes added by spreadsheet editors
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


if __name__ == "__main__":
    data = load_data()
    if data is not None:
        print("Data loaded successfully!")
        print(f"Shape of the dataset: {data.shape}")
        print("Columns:", data.columns.tolist())
        print("First 5 rows:")
        print(data.head())
