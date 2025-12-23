import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import hydra
from omegaconf import DictConfig
import os
from tqdm import tqdm
from collections import defaultdict

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Define Transforms
    # Resize and crop to standard size, convert to Tensor (scales to [0, 1])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # 2. Load Dataset Metadata
    print(f"Loading dataset metadata from {cfg.data_path}...")
    try:
        # We load the full dataset once to get class mappings and file lists
        full_dataset = ImageFolder(root=cfg.data_path, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    num_classes = len(full_dataset.classes)
    print(f"Detected {num_classes} classes.")

    # 3. Group indices by class
    print("Grouping images by class...")
    class_indices = defaultdict(list)
    # Access targets directly if available, otherwise extract from samples
    targets = full_dataset.targets if hasattr(full_dataset, 'targets') else [s[1] for s in full_dataset.samples]
    
    for idx, target in enumerate(targets):
        class_indices[target].append(idx)

    # 4. Process one class at a time
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Computing statistics for each class and saving to '{cfg.output_dir}'...")

    for class_idx in tqdm(range(num_classes), desc="Classes"):
        indices = class_indices[class_idx]
        if not indices:
            continue

        subset = torch.utils.data.Subset(full_dataset, indices)

        dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )

        # --- Streaming accumulators (pixel-wise) ---
        pixel_sum = torch.zeros(3, 224, 224, device=device)
        pixel_sq_sum = torch.zeros(3, 224, 224, device=device)
        num_images = 0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)          # (B, 3, 224, 224)
                pixel_sum += images.sum(dim=0)
                pixel_sq_sum += (images ** 2).sum(dim=0)
                num_images += images.size(0)

        if num_images > 1:
            mean = pixel_sum / num_images
            var = (pixel_sq_sum - num_images * mean ** 2) / (num_images - 1)
            std = torch.sqrt(var.clamp(min=1e-6))
        else:
            mean = torch.zeros(3, 224, 224, device=device)
            std = torch.zeros(3, 224, 224, device=device)

        # --- Save tensors ---
        class_dir = os.path.join(cfg.output_dir, f"class{class_idx + 1}")
        os.makedirs(class_dir, exist_ok=True)
        torch.save(mean.cpu(), os.path.join(class_dir, "mean.pt"))
        torch.save(std.cpu(), os.path.join(class_dir, "std.pt"))

    print("Done.")

if __name__ == "__main__":
    main()
