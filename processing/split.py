import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

csv_dir = "./aggregated_csvs"
output_dir = os.path.join(csv_dir, "splits")
os.makedirs(output_dir, exist_ok=True)

RANDOM_SEED = 4221

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Define data groups
data_groups = {
    "envs": [],
    "rhsfluxes": [],
    "INTrxnfluxes": [],
    "TRANrxnfluxes": [],
}

for fname in sorted(os.listdir(csv_dir)):
    if not fname.endswith(".csv"):
        continue
    fpath = os.path.join(csv_dir, fname)

    if "envs" in fname:
        data_groups["envs"].append(fpath)
    elif "rhsfluxes" in fname:
        data_groups["rhsfluxes"].append(fpath)
    elif "INTrxnfluxes" in fname:
        data_groups["INTrxnfluxes"].append(fpath)
    elif "TRANrxnfluxes" in fname:
        data_groups["TRANrxnfluxes"].append(fpath)

# Helper to read and split one group
def split_group(file_list, group_name):
    if not file_list:
        print(f"No files found for {group_name}")
        return None

    print(f"\n Processing {group_name} ({len(file_list)} files)")
    dfs = [pd.read_csv(f) for f in file_list]
    df = pd.concat(dfs, ignore_index=True)

    # Separate features and labels
    X = df.drop(columns=["label"])
    y = df["label"]

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio),
                                                        stratify=y, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=test_ratio / (test_ratio + val_ratio),
                                                    stratify=y_temp, random_state=RANDOM_SEED)

    # Save splits
    for name, X_part, y_part in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test)
    ]:
        X_part.to_csv(os.path.join(output_dir, f"{group_name}_X_{name}.csv"), index=False)
        y_part.to_csv(os.path.join(output_dir, f"{group_name}_y_{name}.csv"), index=False)
        print(f" Saved {group_name} {name}: {X_part.shape}, labels: {y_part.value_counts().to_dict()}")

    return df

all_flux_df = []
for group_name, file_list in data_groups.items():
    df = split_group(file_list, group_name)
    if df is not None and group_name != "envs":
        all_flux_df.append(df)

if all_flux_df:
    all_flux_df = pd.concat(all_flux_df, ignore_index=True)
    X_all = all_flux_df.drop(columns=["label"])
    y_all = all_flux_df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=(1 - train_ratio),
                                                        stratify=y_all, random_state=RANDOM_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=test_ratio / (test_ratio + val_ratio),
                                                    stratify=y_temp, random_state=RANDOM_SEED)

    for name, X_part, y_part in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test)
    ]:
        X_part.to_csv(os.path.join(output_dir, f"all_fluxes_X_{name}.csv"), index=False)
        y_part.to_csv(os.path.join(output_dir, f"all_fluxes_y_{name}.csv"), index=False)
        print(f" Saved all_fluxes {name}: {X_part.shape}, labels: {y_part.value_counts().to_dict()}")