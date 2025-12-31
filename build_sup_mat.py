import numpy as np
import scipy.io
from collections import defaultdict

META = "/mnt/data/datasets/imagenet/helpers/ILSVRC2012_devkit_t12/data/meta.mat"
WORDNET = "/mnt/data/datasets/imagenet/helpers/ILSVRC2012_devkit_t12/data/wordnet.is_a.txt"

# ------------------------------------------------
# 1. Load ImageNet class → WNID
# ------------------------------------------------
meta = scipy.io.loadmat(META, squeeze_me=True, struct_as_record=False)
synsets = meta["synsets"]

idx_to_wnid = {}
for s in synsets:
    if hasattr(s, "ILSVRC2012_ID"):
        i = int(s.ILSVRC2012_ID)
        if 1 <= i <= 1000:
            idx_to_wnid[i-1] = s.WNID

# ------------------------------------------------
# 2. Load full WordNet hypernym graph
# ------------------------------------------------
parents = defaultdict(list)

with open(WORDNET) as f:
    for line in f:
        c,p = line.strip().split()
        parents[c].append(p)

# ------------------------------------------------
# 3. True WordNet depth via hypernym paths
# ------------------------------------------------
depth = {}

def compute_depth(node):
    if node in depth:
        return depth[node]
    if node not in parents or len(parents[node]) == 0:
        depth[node] = 0
        return 0
    d = 1 + max(compute_depth(p) for p in parents[node])
    depth[node] = d
    return d

for i in range(1000):
    compute_depth(idx_to_wnid[i])

# ------------------------------------------------
# 4. Depth-7 superclass assignment
# ------------------------------------------------
CUT_DEPTH = 7

def nearest_cut_ancestor(node):
    cur = node
    while compute_depth(cur) > CUT_DEPTH:
        cur = max(parents[cur], key=lambda p: compute_depth(p))
    return cur

superclass = {i: nearest_cut_ancestor(idx_to_wnid[i]) for i in range(1000)}

# ------------------------------------------------
# 5. Build equivalence matrix
# ------------------------------------------------
unique = sorted(set(superclass.values()))
sc_to_id = {sc:i for i,sc in enumerate(unique)}
sc_id = np.array([sc_to_id[superclass[i]] for i in range(1000)])

print("Number of superclasses:", len(unique))

M = (sc_id[:,None] == sc_id[None,:]).astype(np.uint8)
np.save("imagenet_depth7_equivalence.npy", M)

print("Saved imagenet_depth7_equivalence.npy")
