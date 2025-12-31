import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = "results_imagenet_stats/plots"
os.makedirs(OUT_DIR, exist_ok=True)

def load_all(norm):
    geom = np.load(f"results_imagenet_stats/imagenet_hist_geometry_{norm}.npy", allow_pickle=True).item()
    metrics = np.load(f"results_imagenet_stats/imagenet_geometry_hist_metrics_{norm}.npy", allow_pickle=True).item()
    return geom, metrics


# ============================================================
# 1. Inner vs Outer histogram
# ============================================================

def plot_histograms(norm, class_id):
    geom, _ = load_all(norm)
    inner = geom[class_id]["inner_vals"]
    outer = geom[class_id]["outer_vals"]

    plt.figure(figsize=(6,4))
    plt.hist(inner, bins=100, density=True, alpha=0.5, label="Same class")
    plt.hist(outer, bins=100, density=True, alpha=0.5, label="Different class")
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title(f"{norm.upper()} distance distributions – class {class_id}")
    plt.legend()
    plt.tight_layout()

    fname = f"{OUT_DIR}/{norm}_hist_class{class_id}.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print("Saved:", fname)


# ============================================================
# 2. Global mean ± std / min / max
# ============================================================

def plot_global_stats(norm):
    _, metrics = load_all(norm)
    classes = sorted(metrics.keys())

    mean_in  = np.array([metrics[c]["mean_inner"] for c in classes])
    mean_out = np.array([metrics[c]["mean_outer"] for c in classes])
    std_in   = np.array([metrics[c]["std_inner"] for c in classes])
    std_out  = np.array([metrics[c]["std_outer"] for c in classes])

    labels = ["Inner", "Outer"]
    means  = [mean_in.mean(), mean_out.mean()]
    stds   = [std_in.mean(),  std_out.mean()]
    mins   = [mean_in.min(),  mean_out.min()]
    maxs   = [mean_in.max(),  mean_out.max()]

    x = np.arange(2)
    plt.figure(figsize=(4,4))
    plt.errorbar(x, means, yerr=stds, fmt='o', capsize=6, label="Mean ± Std", color="black")
    plt.scatter(x, mins, c="red", label="Min")
    plt.scatter(x, maxs, c="green", label="Max")
    plt.xticks(x, labels)
    plt.ylabel("Distance")
    plt.title(f"{norm.upper()} global mean ± std (1000 classes)")
    plt.legend()
    plt.tight_layout()

    fname = f"{OUT_DIR}/{norm}_global_stats.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print("Saved:", fname)


# ============================================================
# 3. ROC AUC distribution
# ============================================================

def plot_global_auc(norm):
    _, metrics = load_all(norm)
    auc = np.array([metrics[c]["roc_auc"] for c in metrics])
    print(f"{norm.upper()} ROC AUC – max : {auc.max():.4f} in class {np.argmax(auc)}")
    print(f"{norm.upper()} ROC AUC – min : {auc.min():.4f} in class {np.argmin(auc)}")
    print(f"{norm.upper()} ROC AUC – mean : {auc.mean():.4f}")

    plt.figure(figsize=(5,4))
    plt.hist(auc, bins=50)
    plt.axvline(0.5)
    plt.xlabel("ROC AUC")
    plt.ylabel("Class count")
    plt.title(f"{norm.upper()} ROC AUC distribution")
    plt.tight_layout()

    fname = f"{OUT_DIR}/{norm}_auc_hist.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print("Saved:", fname)


# ============================================================
# 4. Bayes overlap vs ROC AUC
# ============================================================

def plot_overlap_vs_auc(norm):
    _, metrics = load_all(norm)
    auc   = np.array([metrics[c]["roc_auc"] for c in metrics])
    bayes = np.array([metrics[c]["bayes_overlap"] for c in metrics])

    plt.figure(figsize=(5,5))
    plt.scatter(auc, bayes, s=10)
    plt.xlabel("ROC AUC")
    plt.ylabel("Bayes overlap")
    plt.title(f"{norm.upper()} Bayes overlap vs ROC AUC")
    plt.tight_layout()

    fname = f"{OUT_DIR}/{norm}_overlap_vs_auc.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print("Saved:", fname)


# ============================================================
# 5. ECDF overlay
# ============================================================

def plot_ecdf(norm, class_id):
    geom, _ = load_all(norm)
    inner = np.sort(geom[class_id]["inner_vals"])
    outer = np.sort(geom[class_id]["outer_vals"])

    p = np.linspace(0,1,len(inner))
    q = np.linspace(0,1,len(outer))

    plt.figure(figsize=(6,4))
    plt.plot(inner, p, label="Same class")
    plt.plot(outer, q, label="Different class")
    plt.xlabel("Distance")
    plt.ylabel("ECDF")
    plt.title(f"{norm.upper()} ECDF – class {class_id}")
    plt.legend()
    plt.tight_layout()

    fname = f"{OUT_DIR}/{norm}_ecdf_class{class_id}.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print("Saved:", fname)

if __name__ == "__main__":
    norms = ["l1", "l2", "linf"]

    for norm in norms:
        # 1. Plot histograms and ECDF for selected classes
        
        plot_histograms(norm, 457)
        plot_ecdf(norm, 457)

        # 2. Global mean ± std / min / max
        plot_global_stats(norm)

        # 3. ROC AUC distribution
        plot_global_auc(norm)

        # 4. Bayes overlap vs ROC AUC
        plot_overlap_vs_auc(norm)
