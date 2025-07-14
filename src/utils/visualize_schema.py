

import os
import pandas as pd

def visualize_parquet_structure(file_path: str):
    """
    Reads a Parquet file and prints its structure, including columns,
    data types, and non-null counts.

    Args:
        file_path (str): The full path to the Parquet file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        df = pd.read_parquet(file_path)
        print(f"--- Structure for: {os.path.basename(file_path)} ---")
        df.info()
        print("\n"+"="*50+"\n")
    except Exception as e:
        print(f"Error reading or processing {file_path}: {e}")

def main():
    """
    Main function to locate and visualize all Parquet files in the
    processed data directory.
    """
    # Assumes the script is run from the root of the project directory
    processed_dir = os.path.join("data", "processed")

    if not os.path.isdir(processed_dir):
        print(f"Error: Processed data directory not found at '{processed_dir}'")
        print("Please run the data processing script first.")
        return

    print(f"Scanning for Parquet files in: {os.path.abspath(processed_dir)}\n")

    found_files = False
    for filename in sorted(os.listdir(processed_dir)):
        if filename.endswith(".parquet"):
            found_files = True
            file_path = os.path.join(processed_dir, filename)
            visualize_parquet_structure(file_path)

    if not found_files:
        print("No Parquet files found in the processed data directory.")

if __name__ == "__main__":
    main()

