# import torch

# x = torch.load("results_imagenet_stats/val_dist_matrix_linf.pt")

# print(x)
# print(x.shape)
# print(x.mean())
# print(x.std())
# print(x.min())
# print(x.max())
# print((x==x.T).all().item())

# # Check if the diagonal is very close to zero
# diag = torch.diag(x)
# tolerance = 1e-6  # Define a small tolerance
# is_diag_close_to_zero = torch.all(torch.abs(diag) < tolerance)
# print("Is the diagonal very close to zero?", is_diag_close_to_zero.item())

# ------------ superclass ----------------
import numpy as np
y = np.load("/home/tomer_a/Documents/KDE-analysis-on-imagenet/imagenet_depth7_equivalence.npy")
print (y)
print(y.shape)
print(np.unique(y, axis=0))
print(np.unique(y, axis=0).shape)

def check_superclass_matrix(M):
    # 1. Reflexive: Every class is in the same superclass as itself (Diagonal must be 1)
    is_reflexive = np.all(np.diag(M) == 1)
    
    # 2. Symmetric: If A is with B, B is with A (Matrix equals its transpose)
    is_symmetric = np.all(M == M.T)
    
    # 3. Transitive: If A with B and B with C, A must be with C
    # We use M > 0 to handle cases where M might be int/float rather than strictly bool
    M_bool = M > 0
    is_transitive = not np.any((M_bool @ M_bool) & (~M_bool))
    
    return {
        "Reflexive": is_reflexive,
        "Symmetric": is_symmetric,
        "Transitive": is_transitive,
        "Is_Valid_Equivalence": is_reflexive and is_symmetric and is_transitive
    }

# Usage
results = check_superclass_matrix(y)
print(results)

import numpy as np

# 1. Identify where paths of length 2 exist (indirect connections)
# matrix @ matrix calculates the number of 2-step paths between i and j
indirect_connections = (y @ y) > 0

# 2. Find where an indirect connection exists BUT a direct connection does not
# This is the definition of a transitivity violation
violation_mask = indirect_connections & (y == 0)
# 3. Get the indices of the classes involved
rows, cols = np.where(violation_mask)

# Zip them together to get a list of (i, j) pairs
violating_pairs = list(zip(rows, cols))

print(f"Total violations found: {len(violating_pairs)}")

# 4. (Optional) Print the first few violations and finding the "Bridge" class
# The "bridge" is the class 'k' that connects i and j (i -> k -> j)
if len(violating_pairs) > 0:
    print("\nFirst 5 violations and their bridge classes:")
    for i, j in violating_pairs[:5]:
        # Find which class 'k' connects both i and j
        # We look for index k where matrix[i][k] is 1 AND matrix[k][j] is 1
        bridge_classes = np.where((y[i] == 1) & (y[:, j] == 1))[0]
        
        print(f"Class {i} should be connected to Class {j}")
        print(f"  Reason: They are both connected via Class {bridge_classes}")
        
print