import numpy as np
import torch
import tqdm
import os
from scipy.stats import wasserstein_distance, rv_histogram
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

# ============================================================
# HISTOGRAM WRAPPER (CPU ONLY)
# ============================================================

class HistWrapper:
    def __init__(self, data, bins='fd'):
        """
        Creates a distribution from data using a histogram.
        bins='fd' uses the Freedman-Diaconis rule to automatically 
        select good bin widths.
        """
        # 1. Compute Histogram (Density = True is crucial)
        self.counts, self.edges = np.histogram(data, bins=bins, density=True)
        
        # 2. Create SciPy distribution object for sampling
        self.dist = rv_histogram((self.counts, self.edges))
        
        # 3. Cache bounds for safe log_prob
        self.min_val = self.edges[0]
        self.max_val = self.edges[-1]

    def sample(self, n):
        """Generate n samples from the histogram approximation"""
        return self.dist.rvs(size=n)

    def log_prob(self, x):
        """
        Return log(p(x)). 
        Adds epsilon to prevent log(0) = -inf if x is outside bins.
        """
        # pdf returns 0 outside bounds, so we clamp to 1e-12
        p = self.dist.pdf(x)
        return np.log(np.maximum(p, 1e-12))

# ============================================================
# BUILD + SAVE GEOMETRY
# ============================================================

def build_and_save_geometry(dist_mat_path, save_path,
                            num_classes=1000, imgs_per_class=50,
                            inner_cap=5000, outer_cap=20000): # Caps can be higher now, hists are cheap

    # Load on CPU directly
    D = torch.load(dist_mat_path, map_location="cpu").numpy()
    store = {}

    print("Building histograms...")
    for c in tqdm.tqdm(range(num_classes)):
        s, e = c*imgs_per_class, (c+1)*imgs_per_class

        # Extract data
        inner = D[s:e, s:e][np.triu_indices(imgs_per_class, 1)]
        
        rows = D[s:e]
        outer = np.concatenate([rows[:, :s], rows[:, e:]], 1).ravel()

        # We store the raw data needed to recreate the histogram later
        # (or you could store the counts/edges directly, but saving raw allows re-binning)
        store[c] = {
            "inner_vals": inner, # Keep raw for exact Min/Max/Mean stats
            "outer_vals": outer
        }

    np.save(save_path, store)
    print("Saved Histogram geometry →", save_path)

# ============================================================
# LOAD GEOMETRY
# ============================================================

def load_geometry(path):
    raw = np.load(path, allow_pickle=True).item()
    hist_in, hist_out, inner_cache, outer_cache = {}, {}, {}, {}
    
    print("Loading and fitting histograms...")
    for c, d in tqdm.tqdm(raw.items()):
        # Fit histograms on the fly
        hist_in[c]  = HistWrapper(d["inner_vals"], bins='fd')
        hist_out[c] = HistWrapper(d["outer_vals"], bins='fd')
        
        inner_cache[c] = d["inner_vals"]
        outer_cache[c] = d["outer_vals"]
        
    return hist_in, hist_out, inner_cache, outer_cache

# ============================================================
# METRICS
# ============================================================

def compute_metrics(hist_in, hist_out, inner_cache, outer_cache,
                    mc=20000, bins=256):

    res = {}
    print("Computing metrics...")
    for c in tqdm.tqdm(hist_in):

        # 1. Sample from Histogram Distribution
        xin_np  = hist_in[c].sample(mc)
        xout_np = hist_out[c].sample(mc)

        # Wasserstein
        w = wasserstein_distance(xin_np, xout_np)

        # JS (Discrete binning on samples, as requested)
        lo, hi = min(xin_np.min(), xout_np.min()), max(xin_np.max(), xout_np.max())
        edges = np.linspace(lo, hi, bins+1)
        p, _ = np.histogram(xin_np, edges, density=True)
        q, _ = np.histogram(xout_np, edges, density=True)
        
        # Safety normalization
        p = p / p.sum()
        q = q / q.sum()
        js = jensenshannon(p, q)**2

        # ROC AUC / Mann-Whitney
        u_stat, p_val = mannwhitneyu(xin_np, xout_np, alternative="less")
        auc = roc_auc_score(np.concatenate([np.ones(mc), np.zeros(mc)]), -np.concatenate([xin_np, xout_np]))


        # Bayes Overlap
        # Use a grid covering both supports
        lo2 = min(inner_cache[c].min(), outer_cache[c].min())
        hi2 = max(inner_cache[c].max(), outer_cache[c].max())
        grid = np.linspace(lo2, hi2, 2000)
        
        pin  = np.exp(hist_in[c].log_prob(grid))
        pout = np.exp(hist_out[c].log_prob(grid))
        
        bayes = np.trapz(np.minimum(pin, pout), grid)

        res[c] = {
            "wasserstein": w, "js": js, "roc_auc": auc,
            "mann_whitney_u": u_stat, "mann_whitney_p": p_val,
            "bayes_overlap": bayes,
            "mean_inner": inner_cache[c].mean(), "mean_outer": outer_cache[c].mean(),
            "std_inner": inner_cache[c].std(), "std_outer": outer_cache[c].std()
        }

    return res

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Ensure output dir exists
    if not os.path.exists("results_imagenet_stats"):
        os.makedirs("results_imagenet_stats")

    for norm in ["l1", "l2", "linf"]:
        
        dist_path = f"results_imagenet_stats/val_dist_matrix_{norm}.pt"
        geom_path = f"results_imagenet_stats/imagenet_hist_geometry_{norm}.npy"
        out_path  = f"results_imagenet_stats/imagenet_geometry_hist_metrics_{norm}.npy"

        # Check if input exists
        if not os.path.exists(dist_path):
            print(f"Skipping {norm}, file not found: {dist_path}")
            continue

        # 1. Build Geometry (Histograms)
        if not os.path.exists(geom_path):
            build_and_save_geometry(dist_path, geom_path)

        # 2. Load & Compute
        h_in, h_out, in_cache, out_cache = load_geometry(geom_path)
        metrics = compute_metrics(h_in, h_out, in_cache, out_cache)
        
        np.save(out_path, metrics)
        print(f"Finished {norm} -> Saved to {out_path}")
        keys = list(next(iter(metrics.values())).keys())   # metric names
        classes = sorted(metrics.keys())

        # shape: (num_classes, num_metrics)
        M = np.array([[metrics[c][k] for k in keys] for c in classes], dtype=float)

        mean_by_metric = dict(zip(keys, M.mean(axis=0)))
        std_by_metric  = dict(zip(keys, M.std(axis=0)))

        print("Mean across classes:")
        for k, v in mean_by_metric.items():
            print(f"{k:16s}: {v:.6g}")

        print("\nStd across classes:")
        for k, v in std_by_metric.items():
            print(f"{k:16s}: {v:.6g}")