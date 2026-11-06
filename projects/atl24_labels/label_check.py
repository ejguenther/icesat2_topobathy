from pathlib import Path
import pandas as pd

# Set the path to your folder containing the CSVs
# (Use '.' for the current directory, or provide the full path like 'C:/data' or '/user/data')
FOLDER_PATH = Path("/home/ejg2736/network_drives/walker/exports/nfs_share/Data/ATL24/training_data/labeled")

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