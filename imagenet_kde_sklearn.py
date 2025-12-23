import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for data loading")

    # 1. Define Transforms
    # Resize and crop to standard size, convert to Tensor (scales to [0, 1])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # 2. Load Dataset
    print(f"Loading dataset from {cfg.data_path}...")
    try:
        dataset = ImageFolder(root=cfg.data_path, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, # Shuffle to ensure random sampling if we break early, though we iterate all
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 3. Collect Samples
    # We cannot fit KDE on billions of pixels. We sample 'num_samples' pixels uniformly across the dataset.
    total_batches = len(dataloader)
    samples_per_batch = max(1, int(cfg.num_samples / total_batches))
    
    print(f"Collecting approx {cfg.num_samples} pixels from {len(dataset)} images...")
    
    pixel_samples = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            # Flatten batch to 1D array of pixels
            # images: [B, C, H, W] -> [B*C*H*W]
            batch_pixels = images.view(-1)
            
            # Randomly sample pixels from this batch
            # We use torch.randperm for sampling without replacement within the batch
            if batch_pixels.numel() > samples_per_batch:
                indices = torch.randperm(batch_pixels.numel())[:samples_per_batch]
                selected = batch_pixels[indices]
            else:
                selected = batch_pixels
            
            pixel_samples.append(selected.cpu().numpy())

    # Concatenate all samples
    X_train = np.concatenate(pixel_samples)
    # Reshape for sklearn: (n_samples, n_features) where n_features is 1
    X_train = X_train[:, np.newaxis]
    
    print(f"Fitting KernelDensity model on {X_train.shape[0]} samples with bandwidth={cfg.bandwidth}...")
    
    # 4. Fit KDE
    kde = KernelDensity(bandwidth=cfg.bandwidth, kernel=cfg.kernel)
    kde.fit(X_train)
    
    # 5. Evaluate and Plot
    print("Evaluating density on grid...")
    X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_plot[:, 0], np.exp(log_dens), color='darkorange', lw=2, linestyle='-', label=f"KDE ({cfg.kernel}, bw={cfg.bandwidth})")
    plt.fill_between(X_plot[:, 0], np.exp(log_dens), alpha=0.2, color='darkorange')
    
    plt.title(f'Pixel Intensity KDE (sklearn) - Sampled from {len(dataset)} Images')
    plt.xlabel('Pixel Intensity [0, 1]')
    plt.ylabel('Density')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print(f"Saving plot to {cfg.output_plot}...")
    plt.savefig(cfg.output_plot)

if __name__ == "__main__":
    main()