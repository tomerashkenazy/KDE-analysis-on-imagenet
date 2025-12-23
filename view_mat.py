
import torch

x = torch.load("/home/tomer_a/Documents/KDE-analysis-on-imagenet/results_imagenet_stats/large_matrix.pt")

print(x)
print(x.shape)
print(x.mean())
print(x.std())
print(x.min())
print(x.max())
print(x.median())
