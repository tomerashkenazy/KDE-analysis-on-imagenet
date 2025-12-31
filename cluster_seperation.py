import torch
import numpy as np
import os
import logging
from torchvision import datasets, transforms

# ----------------------------
# Logging
# ----------------------------
log_file = "cluster_separation.log"
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", log_file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger()


# ----------------------------
# Superclass building (connected components)
# ----------------------------
def build_super_labels(labels, super_adj):
    super_adj = super_adj.astype(bool)
    fine_classes = np.unique(labels)

    visited = set()
    super_map = {}
    sid = 0

    for k in fine_classes:
        if k in visited:
            continue

        stack = [int(k)]
        comp = set()

        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            stack.extend(np.where(super_adj[u])[0].tolist())

        for g in comp:
            super_map[g] = sid

        visited |= comp
        sid += 1

    return np.array([super_map[int(y)] for y in labels], dtype=np.int32), sid


# ----------------------------
# Class summaries from distance matrix
# ----------------------------
def compute_class_summaries(dist, labels):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    classes.sort()
    K = len(classes)

    idx_list = [np.where(labels == c)[0] for c in classes]

    A = np.zeros(K)
    diam = np.zeros(K)
    B = np.full((K, K), np.inf)
    minsep = np.full((K, K), np.inf)

    for k in range(K):
        I = idx_list[k]
        block = dist[np.ix_(I, I)]
        n = len(I)
        if n > 1:
            A[k] = (block.sum() - np.trace(block)) / (n * (n - 1))
            diam[k] = block.max()

    for i in range(K):
        Ii = idx_list[i]
        for j in range(i + 1, K):
            Ij = idx_list[j]
            block = dist[np.ix_(Ii, Ij)]
            B[i, j] = B[j, i] = block.mean()
            minsep[i, j] = minsep[j, i] = block.min()

    return classes, idx_list, A, diam, B, minsep


# ----------------------------
# Metrics
# ----------------------------
def silhouette_from_means(A, B):
    """
    Silhouette Score (global average)

    What it measures:
        How well each image fits inside its true class compared to the nearest other class.
        It is a local margin score that detects boundary ambiguity and class overlap.

    Mathematical definition (for each sample i):
        a(i) = mean distance to other samples in its class
        b(i) = minimum mean distance to any other class

        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Global Silhouette = mean_i s(i)

    Value range:
        [-1, 1]
        1   → strong separation
        0   → on class boundary
        <0  → likely misclassified / overlapping

    Interpretation:
        Higher is better.
        Measures *average* separability (not worst-case).
        
man withney u
    """
    a = A
    b = np.min(B, axis=1)
    return float(np.mean((b - a) / np.maximum(a, b)))


def dunn_from_summaries(diam, minsep):
    

    """
    Dunn Index (worst-case class margin)

    What it measures:
        The minimum separation between any two different classes,
        normalized by the maximum internal spread of any class.
        It exposes the single worst overlap in the entire dataset.

    Mathematical definition:
        Δ_k   = max_{i,j in C_k} D(i,j)        (class diameter)
        δ_k,l = min_{i in C_k, j in C_l} D(i,j)

        Dunn = (min_{k≠l} δ_k,l) / (max_k Δ_k)

    Value range:
        (0, ∞)
        <1   → overlapping classes
        ≈1   → touching
        >1   → fully separated

    Interpretation:
        Higher is better.
        Measures *worst-case safety margin* of the representation.
    """

    return float(np.min(minsep) / np.max(diam))


def davies_bouldin_from_means(A, B):
    
    """
    Davies–Bouldin Index (global confusion pressure)

    What it measures:
        For each class, how badly it overlaps with its most similar neighboring class.
        It averages the *worst confusion per class*.

    Mathematical definition:
        A_k = mean intra-class distance of class k
        B_k,l = mean inter-class distance between class k and l

        DB_k = max_{l≠k} (A_k + A_l) / B_k,l
        DB   = mean_k DB_k

    Value range:
        (0, ∞)
        0    → perfect separation
        larger values → worse overlap

    Interpretation:
        Lower is better.
        Measures how many classes are geometrically entangled.
    """

    R = (A[:, None] + A[None, :]) / B
    np.fill_diagonal(R, -np.inf)
    return float(np.mean(np.max(R, axis=1)))


def calinski_harabasz_from_means(A, B, sizes):
    
    """
    Calinski–Harabasz Index (global representation quality)

    What it measures:
        Ratio of total between-class dispersion to total within-class dispersion.
        Indicates how much of the dataset geometry is explained by class structure.

    Mathematical definition:
        W = sum over all intra-class pairwise distances
        B = sum over all inter-class pairwise distances

        CH = (B / (K - 1)) / (W / (N - K))

    Value range:
        (0, ∞)
        Larger values → better defined global class structure

    Interpretation:
        Higher is better.
        Measures overall global embedding quality. """
    
    K = len(sizes)
    N = sum(sizes)

    W = sum(sizes[k] * (sizes[k] - 1) * A[k] for k in range(K))
    Btot = sum(
        2 * sizes[i] * sizes[j] * B[i, j]
        for i in range(K) for j in range(i + 1, K)
    )

    return (Btot / (K - 1)) / (W / (N - K))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    
    eval_dir = "/mnt/data/datasets/imagenet/val/"
    norms = ['1', '2', 'inf']
    # superclass_path = "/home/tomer_a/Documents/epsilon_bounded_contstim/utils/adjacency_matrix.npy"

    logger.info("Loading ImageNet validation labels...")
    dataset_eval = datasets.ImageFolder(
        root=eval_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    labels = np.array([s[1] for s in dataset_eval.samples], dtype=np.int32)

    logger.info("Loading distance matrix...")
    for norm in norms:
        dist_path = f"results_imagenet_stats/val_dist_matrix_l{norm}.pt"
        dist = torch.load(dist_path).cpu().numpy()

        # ----- Fine classes -----
        logger.info("Computing fine class summaries...")
        classes, idx_list, A, diam, B, minsep = compute_class_summaries(dist, labels)
        sizes = [len(I) for I in idx_list]

        logger.info(f"Silhouette norm {norm}: {silhouette_from_means(A, B)}")
        logger.info(f"Dunn norm {norm}: {dunn_from_summaries(diam, minsep)}")
        logger.info(f"Davies–Bouldin norm {norm}: {davies_bouldin_from_means(A, B)}")
        logger.info(f"Calinski–Harabasz norm {norm}: {calinski_harabasz_from_means(A, B, sizes)}")

    # # ----- Superclasses -----
    # logger.info("Loading superclass adjacency...")
    # super_adj = np.load(superclass_path)
    # super_labels, num_super = build_super_labels(labels, super_adj)
    # logger.info(f"Built {num_super} superclasses.")

    # logger.info("Computing superclass summaries...")
    # s_classes, s_idx_list, s_A, s_diam, s_B, s_minsep = compute_class_summaries(dist, super_labels)
    # s_sizes = [len(I) for I in s_idx_list]

    # logger.info(f"Super Silhouette: {silhouette_from_means(s_A, s_B)}")
    # logger.info(f"Super Dunn: {dunn_from_summaries(s_diam, s_minsep)}")
    # logger.info(f"Super Davies–Bouldin: {davies_bouldin_from_means(s_A, s_B)}")
    # logger.info(f"Super Calinski–Harabasz: {calinski_harabasz_from_means(s_A, s_B, s_sizes)}")



