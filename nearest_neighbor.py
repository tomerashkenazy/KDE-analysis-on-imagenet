import torch
import numpy as np
from torchvision import datasets, transforms
import logging
import os
import json

# Set up logging
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "NN_classification.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def nn_classification(
    eval_dir,
    num_images=None,
    dist_matrix_path="results_imagenet_stats/val_dist_matrix_REAL.pt",
    superclass_mat_path="/home/tomer_a/Documents/epsilon_bounded_contstim/utils/adjacency_matrix.npy",
    k_list=[1]
):

    ############################################################################
    # 1) LOAD DATASET + PREPARE LABELS
    ############################################################################
    logger.info("Building dataset...")

    dataset_eval = datasets.ImageFolder(
        root=eval_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )

    if num_images is not None:
        dataset_eval.samples = dataset_eval.samples[:num_images]

    N = len(dataset_eval)
    logger.info(f"Dataset has {N} images.")

    ############################################################################
    # 2) LOAD DISTANCE MATRIX & COMPUTE NN
    ############################################################################
    logger.info(f"Loading distance matrix from {dist_matrix_path}...")
    dist_matrix = torch.load(dist_matrix_path)

    if num_images is not None:
        dist_matrix = dist_matrix[:num_images, :num_images]

    # Ensure dataset matches matrix size (in case matrix was computed on a subset)
    assert dist_matrix.shape[0] == len(dataset_eval), "Distance matrix is not the same size as dataset."

    # Extract targets from the dataset samples
    targets = torch.tensor([s[1] for s in dataset_eval.samples], device=dist_matrix.device)

    # Mask the diagonal (distance to self is 0)
    dist_matrix.fill_diagonal_(float('inf'))

    # Load adjacency matrix (1000, 1000)
    adj_mat = np.load(superclass_mat_path)
    adj_tensor = torch.from_numpy(adj_mat).to(dist_matrix.device)

    # Find the index of the nearest neighbor (max k)
    max_k = max(k_list)
    logger.info(f"Finding top-{max_k} nearest neighbors...")
    _, topk_indices = torch.topk(dist_matrix, max_k, dim=1, largest=False)

    results_list = []

    for k in k_list:
        # Calculate accuracy
        current_indices = topk_indices[:, :k]
        predicted_labels = targets[current_indices]
        correct = (predicted_labels == targets.unsqueeze(1)).any(dim=1).sum().item()
        accuracy = correct / N

        logger.info(f"Top-{k} NN Accuracy: {accuracy * 100:.2f}%")

        # Superclass Accuracy
        is_same_superclass = adj_tensor[targets.unsqueeze(1).expand_as(predicted_labels), predicted_labels]
        superclass_correct = (is_same_superclass > 0).any(dim=1).sum().item()
        superclass_acc = superclass_correct / N
        logger.info(f"Top-{k} Superclass NN Accuracy: {superclass_acc * 100:.2f}%")

        results_list.append((k, f"{accuracy * 100:.2f}", f"{superclass_acc * 100:.2f}"))

    return results_list

if __name__ == "__main__":
    norms = ['l1', 'l2', 'linf']
    results = {}
    for norm in norms:
        logger.info(f"Computing distance matrix with p = {norm}")
        results[norm] = nn_classification(
            eval_dir="/mnt/data/datasets/imagenet/val/",
            dist_matrix_path=f"results_imagenet_stats/val_dist_matrix_{norm}.pt",
            k_list=list(range(1, 11))
        )

    print("NN results:", results)

    output_path = "results_imagenet_stats/nn_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved results to {output_path}")