import os
import pandas as pd

# Directory containing your CSV files
csv_dir = "./aggregated_csvs"

# Whether to overwrite or create new files
overwrite = True  # set to False if you want to save as *_trimmed.csv

for fname in sorted(os.listdir(csv_dir)):
    if not fname.endswith(".csv"):
        continue

    csv_path = os.path.join(csv_dir, fname)
    print(f"🧩 Processing {fname} ...")

    # Read CSV normally
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Could not read {fname}: {e}")
        continue

    # Drop the first two rows safely
    if df.shape[0] > 2:
        df = df.iloc[2:].reset_index(drop=True)
    else:
        print(f"⚠️ {fname} has fewer than 3 rows; skipping trim.")
        continue

    # Save file
    if overwrite:
        save_path = csv_path
    else:
        save_path = os.path.join(csv_dir, fname.replace(".csv", "_trimmed.csv"))

    df.to_csv(save_path, index=False)
    print(f"✅ Saved trimmed file: {save_path} (rows left: {df.shape[0]})\n")