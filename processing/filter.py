import os
import pandas as pd
import numpy as np

csv_dir = "./aggregated_csvs"

output_dir = csv_dir  

# Loop through all CSVs
for fname in sorted(os.listdir(csv_dir)):
    if not fname.endswith(".csv"):
        continue

    csv_path = os.path.join(csv_dir, fname)
    print(f"Processing {fname} ...")

    # Load CSV
    data = pd.read_csv(csv_path)
    arr = data.to_numpy()

    nonzero_mask = np.any(arr != 0, axis=1)

    # Filter and save
    filtered = data[nonzero_mask]
    filtered_path = os.path.join(output_dir, fname)

    filtered.to_csv(filtered_path, index=False)
    print(f"Saved {filtered.shape[0]} non-zero rows (from {data.shape[0]}) → {filtered_path}\n")