import os
import pandas as pd
import numpy as np

csv_dir = "./aggregated_csvs"
output_dir = csv_dir  

n_env_cols = 424
n_rxn_cols = 1715

for fname in sorted(os.listdir(csv_dir)):
    if not fname.endswith(".csv"):
        continue

    csv_path = os.path.join(csv_dir, fname)
    print(f"Processing {fname} ...")

    data = pd.read_csv(csv_path, header=None)
    arr = data.to_numpy()

    # Filter out all-zero rows
    nonzero_mask = np.any(arr != 0, axis=1)
    filtered = data[nonzero_mask].reset_index(drop=True)

    # Detect category (env vs flux)
    if "envs" in fname:
        num_cols = min(filtered.shape[1], n_env_cols)
        cols = [f"c{i+1}" for i in range(num_cols)]
    else:
        num_cols = min(filtered.shape[1], n_rxn_cols)
        cols = [f"r{i+1}" for i in range(num_cols)]

    # Trim/pad columns
    filtered = filtered.iloc[:, :num_cols]
    filtered.columns = cols

    if fname.startswith("fcoop_"):
        filtered["label"] = 1
    elif fname.startswith("comp_"):
        filtered["label"] = 0
    else:
        filtered["label"] = np.nan

    # Save (overwrite)
    output_path = os.path.join(output_dir, fname)
    filtered.to_csv(output_path, index=False)

    print(f"Saved {output_path} | shape {filtered.shape} | label={int(filtered['label'].iloc[0]) if not filtered.empty else 'N/A'}\n")