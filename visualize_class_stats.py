import torch
import matplotlib.pyplot as plt
import os
import hydra
from omegaconf import DictConfig
import numpy as np
from hydra.utils import to_absolute_path
from tqdm import tqdm

def visualize_class_stats(class_num: int, output_dir: str):
    """
    Visualizes the mean and standard deviation tensors for a specific class.
    """
    # Handle path resolution (Hydra changes CWD, so we need absolute path relative to original CWD)
    abs_output_dir = to_absolute_path(output_dir)
    
    # Construct paths (folders are named class1, class2, etc.)
    class_folder = os.path.join(abs_output_dir, f"class{class_num}")
    mean_path = os.path.join(class_folder, "mean.pt")
    std_path = os.path.join(class_folder, "std.pt")

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print(f"Error: Could not find data for class {class_num} at {class_folder}")
        print(f"Make sure 'output_dir' points to the location containing the 'classX' folders.")
        return

    mean_tensor = torch.load(mean_path)
    std_tensor = torch.load(std_path)

    # Convert (C, H, W) -> (H, W, C) for matplotlib
    mean_img = mean_tensor.permute(1, 2, 0).numpy()
    std_img = std_tensor.permute(1, 2, 0).numpy()

    # Clip mean to [0, 1] for display safety
    mean_img = np.clip(mean_img, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(mean_img)
    axes[0].set_title(f"Class {class_num} Mean")
    axes[0].axis('off')

    axes[1].imshow(std_img)
    axes[1].set_title(f"Class {class_num} Std Dev")
    axes[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(class_folder, "visualization.png")
    plt.savefig(save_path)
    plt.close(fig)

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    for i in tqdm(range(1, 1001)):
        visualize_class_stats(i, cfg.output_dir)

if __name__ == "__main__":
    main()