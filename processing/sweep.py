import os
import pandas as pd

csv_dir = "./aggregated_csvs"

for fname in sorted(os.listdir(csv_dir)):
    if not fname.endswith(".csv"):
        continue

    csv_path = os.path.join(csv_dir, fname)
    print(f" Processing {fname} ...")

    # Read
    df = pd.read_csv(csv_path)
    if df.shape[0] > 2:
        df = df.iloc[2:].reset_index(drop=True)
    else:
        print(f"{fname} has fewer than 3 rows; skipping trim.")
        continue

    save_path = csv_path

    df.to_csv(save_path, index=False)
    print(f"Saved trimmed file: {save_path} (rows left: {df.shape[0]})\n")
