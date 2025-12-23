import torch

# Make results deterministic
torch.manual_seed(0)

# Create fake image data (like your case)
# N images, CxHxW pixels
N = 5
x = torch.rand(N, 3, 4, 4)  # small for clarity

# -------------------------------
# Mean (all methods should match)
# -------------------------------
mean_torch = torch.mean(x, dim=0)

mean_stream = x.sum(dim=0) / N

# -------------------------------
# Std 1: torch.std (unbiased=True)
# -------------------------------
std_torch = torch.std(x, dim=0)  # unbiased=True by default

# -------------------------------
# Std 2: population std (biased)
# sqrt(E[x^2] - E[x]^2)
# -------------------------------
var_stream_biased = (x ** 2).mean(dim=0) - mean_stream ** 2
std_stream_biased = torch.sqrt(var_stream_biased)

# -------------------------------
# Std 3: streaming std aligned with PyTorch
# sqrt( sum((x - mean)^2) / (N - 1) )
# -------------------------------
sq_diff_sum = ((x - mean_stream) ** 2).sum(dim=0)
var_stream_unbiased = sq_diff_sum / (N - 1)
std_stream_unbiased = torch.sqrt(var_stream_unbiased)

# -------------------------------
# Print comparisons
# -------------------------------
print("=== MEAN COMPARISON ===")
print("Max abs diff (torch vs stream):",
      (mean_torch - mean_stream).abs().max().item())

print("\n=== STD COMPARISON ===")
print("Max abs diff (torch.std vs biased):",
      (std_torch - std_stream_biased).abs().max().item())

print("Max abs diff (torch.std vs unbiased stream):",
      (std_torch - std_stream_unbiased).abs().max().item())

# Show a single pixel explicitly
c, y, x_idx = 0, 0, 0
print("\nExample pixel [channel=0, y=0, x=0]:")
print("mean (torch)          :", mean_torch[c, y, x_idx].item())
print("mean (stream)         :", mean_stream[c, y, x_idx].item())

print("std (torch)           :", std_torch[c, y, x_idx].item())
print("std (stream biased)   :", std_stream_biased[c, y, x_idx].item())
print("std (stream unbiased) :", std_stream_unbiased[c, y, x_idx].item())



x = torch.rand((50000, 50000), dtype=torch.float32)
torch.save(x, "results_imagenet_stats/large_matrix.pt")
print ("Saved large matrix.")