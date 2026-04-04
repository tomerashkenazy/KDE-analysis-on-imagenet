import argparse
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


LOG_PATH = "/home/tomer_a/Documents/KDE-analysis-on-imagenet/logs/val_to_train_class_min_dists.log"


def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def build_transform() -> transforms.Compose:
    # Must be identical for train and val.
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def validate_class_alignment(train_dataset: datasets.ImageFolder, val_dataset: datasets.ImageFolder) -> None:
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and val class mappings differ; cannot compute classwise minima reliably.")
    if len(train_dataset.classes) != 1000:
        raise ValueError(f"Expected 1000 classes, found {len(train_dataset.classes)}.")


def build_class_indices(train_dataset: datasets.ImageFolder, max_train_per_class: int = None):
    targets = train_dataset.targets if hasattr(train_dataset, "targets") else [s[1] for s in train_dataset.samples]
    class_to_indices = defaultdict(list)
    for idx, cls in enumerate(targets):
        class_to_indices[int(cls)].append(idx)

    ordered = []
    num_classes = len(train_dataset.classes)
    for cls in range(num_classes):
        indices = class_to_indices[cls]
        if max_train_per_class is not None:
            indices = indices[:max_train_per_class]
        if len(indices) == 0:
            raise ValueError(f"Class {cls} has no training images after filtering.")
        ordered.append(indices)
    return ordered


def full_feature_min_over_chunk_cdist(val_flat: torch.Tensor, train_flat: torch.Tensor) -> tuple:
    # Full-feature exact norms with cdist; no feature chunking.
    l1 = torch.cdist(val_flat, train_flat, p=1).amin(dim=1)               # [B]
    l2 = torch.cdist(val_flat, train_flat, p=2).amin(dim=1)               # [B]
    linf = torch.cdist(val_flat, train_flat, p=float("inf")).amin(dim=1)  # [B]
    return l1, l2, linf


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute val->train classwise min distances [N_val,1000,3] for L1/L2/Linf."
    )
    parser.add_argument("--val-root", type=str, required=True, help="Path to ImageNet val root")
    parser.add_argument("--train-root", type=str, required=True, help="Path to ImageNet train root")

    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--train-chunk-size", type=int, default=128)
    parser.add_argument(
        "--feature-chunk-size",
        type=int,
        default=150528,
        help="Deprecated and ignored. Distances now use full-feature torch.cdist.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--max-val", type=int, default=None, help="Debug: limit number of val images")
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=None,
        help="Debug: limit train images per class",
    )

    parser.add_argument(
        "--output-mmap",
        type=str,
        default="results_imagenet_stats/val_train_class_min_dists_l1_l2_linf_50000x1000x3.f32.mmap",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        default="results_imagenet_stats/val_train_class_min_dists_l1_l2_linf_50000x1000x3.pt",
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    logger = setup_logging()
    os.makedirs("results_imagenet_stats", exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info("Using device: %s", device)
    logger.info("Building datasets with identical transforms for train and val.")
    logger.info("Distance mode: full-feature torch.cdist for L1/L2/Linf (feature chunking disabled).")

    transform = build_transform()
    val_dataset = datasets.ImageFolder(root=args.val_root, transform=transform)
    train_dataset = datasets.ImageFolder(root=args.train_root, transform=transform)

    validate_class_alignment(train_dataset, val_dataset)

    n_val_total = len(val_dataset)
    n_val = n_val_total if args.max_val is None else min(n_val_total, args.max_val)
    n_classes = len(train_dataset.classes)

    val_indices = list(range(n_val))
    val_subset = Subset(val_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    class_indices = build_class_indices(train_dataset, max_train_per_class=args.max_train_per_class)
    class_counts = [len(idx_list) for idx_list in class_indices]

    logger.info("Val images: %d (from total %d)", n_val, n_val_total)
    logger.info("Classes: %d", n_classes)
    logger.info(
        "Train samples per class: min=%d max=%d mean=%.1f",
        int(np.min(class_counts)),
        int(np.max(class_counts)),
        float(np.mean(class_counts)),
    )

    mmap_dir = os.path.dirname(args.output_mmap)
    if mmap_dir:
        os.makedirs(mmap_dir, exist_ok=True)

    logger.info("Creating memmap at %s", args.output_mmap)
    min_dists_mmap = np.memmap(
        args.output_mmap,
        mode="w+",
        dtype=np.float32,
        shape=(n_val, n_classes, 3),
    )
    min_dists_mmap.fill(np.inf)
    min_dists_mmap.flush()

    val_offset = 0

    with torch.no_grad():
        for val_batch_idx, (val_imgs_cpu, _) in enumerate(tqdm(val_loader, desc="Val batches")):
            val_imgs = val_imgs_cpu.to(device, non_blocking=args.pin_memory)
            val_flat = val_imgs.flatten(1).contiguous()
            bsz = val_flat.shape[0]

            batch_min_l1 = torch.full((bsz, n_classes), float("inf"), device=device)
            batch_min_l2 = torch.full((bsz, n_classes), float("inf"), device=device)
            batch_min_linf = torch.full((bsz, n_classes), float("inf"), device=device)

            for cls in tqdm(range(n_classes), desc=f"Classes@val_batch{val_batch_idx}", leave=False):
                idxs = class_indices[cls]

                for k0 in range(0, len(idxs), args.train_chunk_size):
                    k1 = min(k0 + args.train_chunk_size, len(idxs))
                    chunk_indices = idxs[k0:k1]

                    train_imgs = [train_dataset[i][0] for i in chunk_indices]
                    train_chunk_cpu = torch.stack(train_imgs, dim=0)
                    train_flat = train_chunk_cpu.flatten(1).to(device, non_blocking=args.pin_memory)

                    try:
                        cur_l1, cur_l2, cur_linf = full_feature_min_over_chunk_cdist(val_flat, train_flat)
                    except RuntimeError as e:
                        raise RuntimeError(
                            "Full-feature distance kernel failed. Lower --val-batch-size and/or "
                            "--train-chunk-size. Original error: "
                            f"{e}"
                        ) from e

                    batch_min_l2[:, cls] = torch.minimum(batch_min_l2[:, cls], cur_l2)
                    batch_min_l1[:, cls] = torch.minimum(batch_min_l1[:, cls], cur_l1)
                    batch_min_linf[:, cls] = torch.minimum(batch_min_linf[:, cls], cur_linf)

            batch_np = torch.stack([batch_min_l1, batch_min_l2, batch_min_linf], dim=2).cpu().numpy()
            min_dists_mmap[val_offset:val_offset + bsz, :, :] = batch_np
            min_dists_mmap.flush()

            logger.info(
                "Finished val batch %d: wrote rows [%d, %d)",
                val_batch_idx,
                val_offset,
                val_offset + bsz,
            )
            val_offset += bsz

    logger.info("Converting memmap to torch tensor and saving .pt to %s", args.output_pt)
    pt_dir = os.path.dirname(args.output_pt)
    if pt_dir:
        os.makedirs(pt_dir, exist_ok=True)

    final_np = np.memmap(
        args.output_mmap,
        mode="r",
        dtype=np.float32,
        shape=(n_val, n_classes, 3),
    )
    final_tensor = torch.from_numpy(np.array(final_np, copy=True))
    torch.save(final_tensor, args.output_pt)

    logger.info("Done. Saved tensor shape %s", tuple(final_tensor.shape))


if __name__ == "__main__":
    main()
