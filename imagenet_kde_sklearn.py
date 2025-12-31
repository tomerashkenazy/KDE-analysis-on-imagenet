import numpy as np
import torch, tqdm, os
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu


# ============================================================
# GPU-SAFE KDE
# ============================================================

class TorchKDE:
    def __init__(self, samples, bandwidth, device="cuda"):
        self.x_cpu = torch.tensor(samples, dtype=torch.float32)
        self.h = bandwidth
        self.device = device
        self.inv_2h2 = 1.0 / (2 * bandwidth * bandwidth)
        self.log_norm = -np.log(self.x_cpu.shape[0] * bandwidth * np.sqrt(2*np.pi))

    def sample(self, n):
        idx = torch.randint(0, self.x_cpu.shape[0], (n,))
        return self.x_cpu[idx] + torch.randn(n) * self.h

    def log_prob(self, y, chunk=2048):
        y = y.to(self.device)
        x = self.x_cpu.to(self.device)
        out = []
        for i in range(0, len(y), chunk):
            yc = y[i:i+chunk][:,None]
            diffs = yc - x[None,:]
            lp = torch.logsumexp(-(diffs**2)*self.inv_2h2, dim=1) + self.log_norm
            out.append(lp)
        del x
        torch.cuda.empty_cache()
        return torch.cat(out)
    


# ============================================================
# BUILD + SAVE KDE GEOMETRY
# ============================================================

def build_and_save_geometry(dist_mat_path, save_path,
                            num_classes=1000, imgs_per_class=50,
                            bw_in=0.03, bw_out=0.04,
                            inner_cap=1000, outer_cap=20000):

    D = torch.load(dist_mat_path).cpu().numpy()
    store = {}

    for c in tqdm.tqdm(range(num_classes)):
        s,e = c*imgs_per_class, (c+1)*imgs_per_class

        inner = D[s:e,s:e][np.triu_indices(imgs_per_class,1)]
        Xin = np.random.choice(inner, min(inner_cap,len(inner)), replace=False)

        rows = D[s:e]
        outer = np.concatenate([rows[:,:s], rows[:,e:]],1).ravel()
        Xout = np.random.choice(outer, outer_cap, replace=False)

        store[c] = {
            "Xin": Xin,
            "Xout": Xout,
            "inner_vals": inner,
            "outer_vals": outer,
            "bw_in": bw_in,
            "bw_out": bw_out
        }

    np.save(save_path, store)
    print("Saved KDE geometry →", save_path)

# ============================================================
# LOAD KDE GEOMETRY
# ============================================================

def load_geometry(path):
    raw = np.load(path, allow_pickle=True).item()
    kde_in, kde_out, inner_cache, outer_cache = {},{},{},{}
    for c,d in raw.items():
        kde_in[c]  = TorchKDE(d["Xin"],  d["bw_in"])
        kde_out[c] = TorchKDE(d["Xout"], d["bw_out"])
        inner_cache[c] = d["inner_vals"]
        outer_cache[c] = d["outer_vals"]
    return kde_in, kde_out, inner_cache, outer_cache

# ============================================================
# METRICS
# ============================================================

def compute_metrics(kde_in, kde_out, inner_cache, outer_cache,
                    mc=20000, bins=256):

    res = {}
    for c in tqdm.tqdm(kde_in):

        xin  = kde_in[c].sample(mc)
        xout = kde_out[c].sample(mc)
        xin_np, xout_np = xin.cpu().numpy(), xout.cpu().numpy()

        # Wasserstein
        w = wasserstein_distance(xin_np, xout_np)

        # JS
        lo,hi = min(xin_np.min(),xout_np.min()), max(xin_np.max(),xout_np.max())
        edges = np.linspace(lo,hi,bins+1)
        p,_ = np.histogram(xin_np,edges,density=True)
        q,_ = np.histogram(xout_np,edges,density=True)
        js = jensenshannon(p,q)**2

        # ROC AUC
        u_stat, p_val = mannwhitneyu(xin_np, xout_np, alternative="less")
        auc = u_stat / (mc * mc)

        # y = np.concatenate([np.ones(mc),np.zeros(mc)])
        # auc = roc_auc_score(y, -np.concatenate([xin_np,xout_np]))
        
        # Mann-Whitney U test (equivalent to ROC AUC)

        # KLs
        kl_io = (kde_in[c].log_prob(xin) - kde_out[c].log_prob(xin)).mean().item()
        kl_oi = (kde_out[c].log_prob(xout) - kde_in[c].log_prob(xout)).mean().item()

        # Bayes overlap
        lo2,hi2 = min(inner_cache[c].min(),outer_cache[c].min()), max(inner_cache[c].max(),outer_cache[c].max())
        grid = torch.linspace(lo2,hi2,2000,device="cuda")
        pin  = torch.exp(kde_in[c].log_prob(grid))
        pout = torch.exp(kde_out[c].log_prob(grid))
        bayes = torch.trapz(torch.minimum(pin,pout),grid).item()

        res[c] = {
            "wasserstein":w,"js":js,"roc_auc":auc,
            "kl_in_out":kl_io,"kl_out_in":kl_oi,"jeffreys":0.5*(kl_io+kl_oi),
            "bayes_overlap":bayes,
            "mean_inner":inner_cache[c].mean(),"mean_outer":outer_cache[c].mean(),
            "std_inner":inner_cache[c].std(),"std_outer":outer_cache[c].std()
        }

    return res

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for norm in ["l1","l2","linf"]:

        dist_path = f"results_imagenet_stats/val_dist_matrix_{norm}.pt"
        geom_path = f"results_imagenet_stats/imagenet_kde_geometry_{norm}.npy"

        if not os.path.exists(geom_path):
            build_and_save_geometry(dist_path, geom_path)

        kde_in,kde_out,inner_cache,outer_cache = load_geometry(geom_path)
        metrics = compute_metrics(kde_in,kde_out,inner_cache,outer_cache)
        np.save(f"results_imagenet_stats/imagenet_geometry_metrics_{norm}.npy", metrics)

        print("Finished", norm)
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
