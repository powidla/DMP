import os
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix

# Path to your local models
local_dir = "/Users/sasha/Desktop/FriendOrFoe/FOFdata/HGT_models_Agora"

def load_mat_file(file_path):
    """Load a .mat file and extract metabolic model + number of reactions."""
    data = loadmat(file_path)
    metab_model = data['MetabModel'][0, 0]

    keys = [
        "bmi", "rhs_ext_lb", "rhs_ext_ub", "rhs_int_lb", "rhs_int_ub",
        "S_ext", "S_int", "S_ext2int", "S_unmapped",
        "lb", "ub", "name"
    ]

    components = {
        key: (csc_matrix(metab_model[i]) if key.startswith("S_") else metab_model[i])
        for i, key in enumerate(keys)
    }

    name = str(components["name"][0])
    n_reactions = components["lb"].shape[0]

    return components, name, n_reactions

# Step 1: List all .mat files in the local directory
mat_files = sorted([os.path.join(local_dir, f) for f in os.listdir(local_dir) if f.endswith(".mat")])
print(f"Found {len(mat_files)} mat files")

# Step 2: Collect reaction counts
reaction_counts = []
for mf in mat_files:
    _, name, n_reactions = load_mat_file(mf)
    reaction_counts.append((mf, name, n_reactions))

reaction_counts = sorted(reaction_counts, key=lambda x: x[2])
counts = [c[2] for c in reaction_counts]

print(f"Min reactions: {counts[0]}, Max: {counts[-1]}, Median: {np.median(counts)}")

# Step 3: Select 3 small, 4 mid, 3 large
n_total = len(reaction_counts)
small = reaction_counts[:3]
mid = reaction_counts[n_total//2 - 2 : n_total//2 + 2]  # 4 around median
large = reaction_counts[-3:]

selected_models = small + mid + large

print("\nSelected Models:")
for mf, name, n_reactions in selected_models:
    print(f"{os.path.basename(mf)} ({name}) -> {n_reactions} reactions")