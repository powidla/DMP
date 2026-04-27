import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

# Root directory containing subfolders AGORA_M446wM1 ... AGORA_M446wM818
root_dir = "./"

categories = [
    "comp_envs", "fcoop_envs",
    "comp_INTrxnfluxes", "fcoop_INTrxnfluxes",
    "comp_rhsfluxes", "fcoop_rhsfluxes",
    "comp_TRANrxnfluxes", "fcoop_TRANrxnfluxes"
]

# Helper: load and extract array from .mat
def load_array(mat_path):
    try:
        data = loadmat(mat_path)
        key = os.path.splitext(os.path.basename(mat_path))[0]  # e.g. comp_INTrxnfluxes_MimaxMifixMj
        arr = data[key]
        if hasattr(arr, "toarray"):
            arr = arr.toarray()  # handle sparse matrices
        return np.array(arr)
    except Exception as e:
        print(f"Could not read {mat_path}: {e}")
        return None

def pad_arrays(arr_list):
    if not arr_list:
        return np.empty((0, 0))
    max_rows = max(a.shape[0] for a in arr_list)
    max_cols = max(a.shape[1] for a in arr_list)
    padded = []
    for a in arr_list:
        pad_r = max_rows - a.shape[0]
        pad_c = max_cols - a.shape[1]
        a_padded = np.pad(a, ((0, pad_r), (0, pad_c)), mode='constant', constant_values=0)
        padded.append(a_padded)
    return np.concatenate(padded, axis=0)

# Main collection dictionary
all_data = {cat: [] for cat in categories}

for folder in sorted(os.listdir(root_dir)):
    subdir = os.path.join(root_dir, folder)
    if not os.path.isdir(subdir) or not folder.startswith("AG_"):
        continue

    print(f"Processing {folder}...")

    for cat in categories:
        matches = [f for f in os.listdir(subdir) if f.startswith(cat) and f.endswith(".mat")]
        for fname in matches:
            mat_path = os.path.join(subdir, fname)
            arr = load_array(mat_path)
            if arr is not None:
                all_data[cat].append(arr)

# Save results
output_dir = os.path.join(root_dir, "aggregated_csvs")
os.makedirs(output_dir, exist_ok=True)
