"""Scan ATL24 training-label CSVs and print the unique 'label' values in each.

Usage: python label_check.py /path/to/ATL24/training_data/labeled
"""
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("folder", type=Path, help="Folder containing the labeled .csv files")
FOLDER_PATH = parser.parse_args().folder

# Find all .csv files in the directory
csv_files = list(FOLDER_PATH.glob("*.csv"))

if not csv_files:
    print(f"No .csv files found in '{FOLDER_PATH.resolve()}'")
else:
    print(f"Scanning {len(csv_files)} files for unique 'label' values...\n")
    print("-" * 50)

    for file_path in csv_files:
        try:
            # usecols=['label'] optimizes performance by only reading that specific column
            df = pd.read_csv(file_path, usecols=["label"])

            # Get unique values and drop any NaNs if present
            unique_labels = df["label"].dropna().unique()

            print(f"File: {file_path.name}")
            print(f"Unique Labels: {list(unique_labels)}")
            print("-" * 50)

        except KeyError:
            print(f"File: {file_path.name} -> ⚠️ 'label' column not found.")
            print("-" * 50)
        except Exception as e:
            print(f"File: {file_path.name} -> ❌ Error reading file: {e}")
            print("-" * 50)