import json
import matplotlib.pyplot as plt

# Load real results
json_path = "/home/tomer_a/Documents/KDE-analysis-on-imagenet/results_imagenet_stats/nn_results.json"
with open(json_path, "r") as f:
    results = json.load(f)

# Chance levels
chance_exact = 0.1          # 1 / 1000 classes
chance_super = 100 / 124    # 1 / 124 superclasses

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

# -------- Exact NN (normalized) --------
for norm, vals in results.items():
    k = [v[0] for v in vals]
    acc = [float(v[1]) for v in vals]
    ax1.plot(k, acc, marker='o', label=norm)

ax1.axhline(chance_exact, color="red", linestyle="--", linewidth=2, label="Chance ")
ax1.set_title("Top-K NN Accuracy (Exact Class)")
ax1.set_xlabel("k")
ax1.set_ylabel("Accuracy")
ax1.set_xticks(k)
ax1.grid(True)
ax1.legend()

# -------- Superclass NN (normalized) --------
for norm, vals in results.items():
    k = [v[0] for v in vals]
    sacc = [float(v[2]) / chance_super for v in vals]
    ax2.plot(k, sacc, marker='o', label=norm)

ax2.axhline(chance_super, color="red", linestyle="--", linewidth=2, label="Chance ")
ax2.set_title("Top-K Superclass NN Accuracy")
ax2.set_xlabel("k")
ax2.set_ylabel("Accuracy")

ax2.set_xticks(k)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig("nn_accuracy_comparison.png")
