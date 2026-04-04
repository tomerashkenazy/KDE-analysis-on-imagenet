import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch


DEFAULT_BASE_EPS = [
    1 / 128,
    1 / 64,
    1 / 32,
    1 / 16,
    1 / 8,
    1 / 4,
    1 / 2,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
]

NORM_TO_INDEX = {"l1": 0, "l2": 1, "linf": 2}


def _extract_from_container(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        if len(obj) == 1:
            return next(iter(obj.values()))
        raise KeyError(f"Key '{key}' not found. Available keys: {list(obj.keys())}")
    return obj


def load_array(path: str, key: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pt":
        obj = torch.load(path, map_location="cpu")
        arr = _extract_from_container(obj, key)
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        else:
            arr = np.asarray(arr)
        return arr

    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr)

    if ext == ".npz":
        obj = np.load(path, allow_pickle=False)
        if key in obj.files:
            return np.asarray(obj[key])
        if len(obj.files) == 1:
            return np.asarray(obj[obj.files[0]])
        raise KeyError(f"Key '{key}' not found. Available keys: {obj.files}")

    raise ValueError(f"Unsupported file extension '{ext}' for path: {path}")


def validate_inputs(min_dists: np.ndarray, labels: np.ndarray) -> Tuple[int, int]:
    if min_dists.ndim != 3:
        raise ValueError(f"min_dists must have shape [N, 1000, 3]. Got {min_dists.shape}")
    n, num_classes, num_norms = min_dists.shape
    if num_classes != 1000 or num_norms != 3:
        raise ValueError(f"min_dists must have shape [N, 1000, 3]. Got {min_dists.shape}")

    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [N]. Got {labels.shape}")
    if labels.shape[0] != n:
        raise ValueError(
            f"labels length must match min_dists N ({n}). Got {labels.shape[0]}"
        )

    if np.any(labels < 0) or np.any(labels >= 1000):
        bad_min = int(labels.min())
        bad_max = int(labels.max())
        raise ValueError(f"labels must be in [0, 999]. Got min={bad_min}, max={bad_max}")

    return n, num_classes


def scale_eps(norm_name: str, base_eps: float) -> float:
    if norm_name == "l2":
        return float(base_eps)
    if norm_name == "linf":
        return float(base_eps) / 255.0
    if norm_name == "l1":
        return float(base_eps) * (255.0 / 2.0)
    raise ValueError(f"Unknown norm: {norm_name}")


def inverse_scale_eps(norm_name: str, effective_eps: float) -> float:
    if norm_name == "l2":
        return float(effective_eps)
    if norm_name == "linf":
        return float(effective_eps) * 255.0
    if norm_name == "l1":
        return float(effective_eps) / (255.0 / 2.0)
    raise ValueError(f"Unknown norm: {norm_name}")


def compute_score_curve_for_norm(
    min_dists_norm: np.ndarray,
    labels: np.ndarray,
    norm_name: str,
    base_eps_values: List[float],
) -> Dict[str, List[float]]:
    n = min_dists_norm.shape[0]
    row_indices = np.arange(n)

    effective_eps_list: List[float] = []
    accuracy_limit_list: List[float] = []
    mean_k_list: List[float] = []
    gt_reachable_fraction_aux_list: List[float] = []

    for base_eps in base_eps_values:
        effective_eps = scale_eps(norm_name, base_eps)
        threshold = 2.0 * effective_eps

        # reachable[i, c] is True when class c is reachable for image i.
        reachable = min_dists_norm <= threshold

        k = reachable.sum(axis=1).astype(np.float64)
        gt_reachable = reachable[row_indices, labels]

        # New definition:
        # - If no class is reachable (K_i=0), score_i = 1.
        # - If no non-GT class is reachable, score_i = 1.
        # - Otherwise, score_i = 1 / K_i.
        gt_reachable_int = gt_reachable.astype(np.int64)
        non_gt_reachable_count = k - gt_reachable_int
        score = np.ones(n, dtype=np.float64)
        ambiguous = non_gt_reachable_count > 0
        score[ambiguous] = 1.0 / k[ambiguous]

        effective_eps_list.append(float(effective_eps))
        accuracy_limit_list.append(float(score.mean()))
        mean_k_list.append(float(k.mean()))
        gt_reachable_fraction_aux_list.append(float(gt_reachable.mean()))

    return {
        "base_eps": [float(x) for x in base_eps_values],
        "effective_eps": effective_eps_list,
        "accuracy_limit": accuracy_limit_list,
        "mean_K": mean_k_list,
        "gt_reachable_fraction_aux": gt_reachable_fraction_aux_list,
    }


def compute_score_curve_for_norm_effective_grid(
    min_dists_norm: np.ndarray,
    labels: np.ndarray,
    norm_name: str,
    effective_eps_values: List[float],
) -> Dict[str, List[float]]:
    base_eps_values = [inverse_scale_eps(norm_name, e) for e in effective_eps_values]

    n = min_dists_norm.shape[0]
    row_indices = np.arange(n)

    accuracy_limit_list: List[float] = []
    mean_k_list: List[float] = []
    gt_reachable_fraction_aux_list: List[float] = []

    for effective_eps in effective_eps_values:
        threshold = 2.0 * float(effective_eps)
        reachable = min_dists_norm <= threshold

        k = reachable.sum(axis=1).astype(np.float64)
        gt_reachable = reachable[row_indices, labels]

        gt_reachable_int = gt_reachable.astype(np.int64)
        non_gt_reachable_count = k - gt_reachable_int
        score = np.ones(n, dtype=np.float64)
        ambiguous = non_gt_reachable_count > 0
        score[ambiguous] = 1.0 / k[ambiguous]

        accuracy_limit_list.append(float(score.mean()))
        mean_k_list.append(float(k.mean()))
        gt_reachable_fraction_aux_list.append(float(gt_reachable.mean()))

    return {
        "base_eps": [float(x) for x in base_eps_values],
        "effective_eps": [float(x) for x in effective_eps_values],
        "accuracy_limit": accuracy_limit_list,
        "mean_K": mean_k_list,
        "gt_reachable_fraction_aux": gt_reachable_fraction_aux_list,
    }


def save_results_json(path: str, results: Dict[str, Dict[str, List[float]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_results_csv(path: str, results: Dict[str, Dict[str, List[float]]]) -> None:
    fieldnames = [
        "norm",
        "base_eps",
        "effective_eps",
        "accuracy_limit",
        "mean_K",
        "gt_reachable_fraction_aux",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for norm in ["l1", "l2", "linf"]:
            vals = results[norm]
            for i in range(len(vals["base_eps"])):
                writer.writerow(
                    {
                        "norm": norm,
                        "base_eps": vals["base_eps"][i],
                        "effective_eps": vals["effective_eps"][i],
                        "accuracy_limit": vals["accuracy_limit"][i],
                        "mean_K": vals["mean_K"][i],
                        "gt_reachable_fraction_aux": vals["gt_reachable_fraction_aux"][i],
                    }
                )


def plot_accuracy_limits(path: str, results: Dict[str, Dict[str, List[float]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, norm in zip(axes, ["l1", "l2", "linf"]):
        x = np.asarray(results[norm]["effective_eps"], dtype=np.float64)
        y = np.asarray(results[norm]["accuracy_limit"], dtype=np.float64) * 100.0

        ax.plot(x, y, marker="o", linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Epsilon (effective)")
        ax.set_title(norm.upper())
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5.0, 105.0)

        # Show human-readable ticks instead of pure 10^k notation.
        if norm == "l2":
            p2 = np.array([2.0 ** k for k in range(-4, 20)], dtype=np.float64)
            shown = p2[(p2 >= float(x.min())) & (p2 <= float(x.max()))]
            if shown.size > 0:
                ax.set_xticks(shown)
                ax.set_xticklabels([f"{v:g}" for v in shown])
            else:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _pos: f"{val:g}"))
        elif norm == "l1":
            mult = 255.0 / 2.0
            p2_scaled = mult * np.array([2.0 ** k for k in range(-4, 24)], dtype=np.float64)
            shown = p2_scaled[(p2_scaled >= float(x.min())) & (p2_scaled <= float(x.max()))]
            if shown.size > 0:
                ax.set_xticks(shown)
                ax.set_xticklabels([f"{v:g}" for v in shown])
            else:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _pos: f"{val:g}"))
        else:
            # Linf in fractions of 1/255 for readability.
            linf_ticks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 255], dtype=np.float64) / 255.0
            tmin, tmax = float(x.min()), float(x.max())
            shown = linf_ticks[(linf_ticks >= tmin) & (linf_ticks <= tmax)]
            if shown.size > 0:
                ax.set_xticks(shown)
                ax.set_xticklabels([f"{int(round(v * 255))}/255" for v in shown])

    axes[0].set_ylabel("Geometry-Based Robust Accuracy Limit (%)")
    fig.suptitle("Robust Accuracy Limit vs Epsilon", fontsize=14)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_summary_table(results: Dict[str, Dict[str, List[float]]]) -> None:
    print("\nSummary (first + last epsilon per norm):")
    header = (
        "norm",
        "base_eps",
        "eff_eps",
        "acc_limit",
        "mean_K",
        "gt_reach_aux",
    )
    print(f"{header[0]:<6} {header[1]:>10} {header[2]:>12} {header[3]:>12} {header[4]:>12} {header[5]:>14}")

    for norm in ["l1", "l2", "linf"]:
        vals = results[norm]
        for i in [0, len(vals["base_eps"]) - 1]:
            print(
                f"{norm:<6} "
                f"{vals['base_eps'][i]:>10.6g} "
                f"{vals['effective_eps'][i]:>12.6g} "
                f"{vals['accuracy_limit'][i]:>12.6g} "
                f"{vals['mean_K'][i]:>12.6g} "
                f"{vals['gt_reachable_fraction_aux'][i]:>14.6g}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute geometry-based robust accuracy limits from precomputed min_dists."
    )
    parser.add_argument(
        "--min-dists-path",
        type=str,
        required=True,
        help="Path to min_dists file (.pt/.npy/.npz), expected shape [N,1000,3].",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        required=True,
        help="Path to labels file (.pt/.npy/.npz), expected shape [N].",
    )
    parser.add_argument(
        "--min-dists-key",
        type=str,
        default="min_dists",
        help="Key to use if min_dists path contains a dict/npz.",
    )
    parser.add_argument(
        "--labels-key",
        type=str,
        default="labels",
        help="Key to use if labels path contains a dict/npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_imagenet_stats",
        help="Output directory for figure and structured results.",
    )
    parser.add_argument(
        "--base-eps",
        type=float,
        nargs="+",
        default=DEFAULT_BASE_EPS,
        help="Base epsilon sweep values.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a short summary table.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix tag appended to output filenames (for versioned outputs).",
    )
    parser.add_argument(
        "--eps-grid-mode",
        type=str,
        choices=["base", "informative"],
        default="base",
        help="base: use base-eps with norm scaling; informative: use norm-specific effective-eps ranges.",
    )
    parser.add_argument("--num-eps-points", type=int, default=15, help="Number of points for informative grid.")
    parser.add_argument("--l1-eps-min", type=float, default=1e3, help="L1 effective epsilon min.")
    parser.add_argument("--l1-eps-max", type=float, default=1e5, help="L1 effective epsilon max.")
    parser.add_argument("--l2-eps-min", type=float, default=10.0, help="L2 effective epsilon min.")
    parser.add_argument("--l2-eps-max", type=float, default=100.0, help="L2 effective epsilon max.")
    parser.add_argument("--linf-eps-min", type=float, default=1e-1, help="Linf effective epsilon min.")
    parser.add_argument("--linf-eps-max", type=float, default=1.0, help="Linf effective epsilon max.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    min_dists = load_array(args.min_dists_path, args.min_dists_key)
    labels = load_array(args.labels_path, args.labels_key)

    min_dists = np.asarray(min_dists, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    n, _ = validate_inputs(min_dists, labels)
    print(f"Loaded min_dists with shape {min_dists.shape} and labels with shape {labels.shape} (N={n}).")

    results: Dict[str, Dict[str, List[float]]] = {}

    for norm_name, idx in [("l1", 0), ("l2", 1), ("linf", 2)]:
        if args.eps_grid_mode == "informative":
            if norm_name == "l1":
                eff = np.logspace(np.log10(args.l1_eps_min), np.log10(args.l1_eps_max), args.num_eps_points)
            elif norm_name == "l2":
                eff = np.logspace(np.log10(args.l2_eps_min), np.log10(args.l2_eps_max), args.num_eps_points)
            else:
                eff = np.logspace(np.log10(args.linf_eps_min), np.log10(args.linf_eps_max), args.num_eps_points)

            results[norm_name] = compute_score_curve_for_norm_effective_grid(
                min_dists_norm=min_dists[:, :, idx],
                labels=labels,
                norm_name=norm_name,
                effective_eps_values=[float(e) for e in eff],
            )
        else:
            results[norm_name] = compute_score_curve_for_norm(
                min_dists_norm=min_dists[:, :, idx],
                labels=labels,
                norm_name=norm_name,
                base_eps_values=args.base_eps,
            )

    tag = f"_{args.output_tag}" if args.output_tag else ""
    json_path = os.path.join(args.output_dir, f"geometry_robust_accuracy_limit_results{tag}.json")
    csv_path = os.path.join(args.output_dir, f"geometry_robust_accuracy_limit_results{tag}.csv")
    fig_path = os.path.join(args.output_dir, f"geometry_robust_accuracy_limit_plot{tag}.png")

    save_results_json(json_path, results)
    save_results_csv(csv_path, results)
    plot_accuracy_limits(fig_path, results)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")

    if args.print_summary:
        print_summary_table(results)


if __name__ == "__main__":
    main()
