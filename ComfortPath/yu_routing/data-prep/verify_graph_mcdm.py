"""
Quick sanity check for MCDM graph attributes.
Run after rebuilding main_graph.pkl.
"""
import pickle
from pathlib import Path
import numpy as np

GRAPH_CACHE_PATH = Path(__file__).resolve().parent / "data" / "main_graph.pkl"

print("Loading graph...")
with open(GRAPH_CACHE_PATH, "rb") as f:
    G = pickle.load(f)

slope_norms = []
length_norms = []
slope_scores = []

for u, v, d in G.edges(data=True):
    slope_norms.append(d.get("slope_norm"))
    length_norms.append(d.get("length_norm"))
    slope_scores.append(d.get("slope_score"))

# Check that _norm columns exist
missing_slope_norm = sum(1 for x in slope_norms if x is None)
missing_length_norm = sum(1 for x in length_norms if x is None)
print(f"\nEdges missing slope_norm:  {missing_slope_norm:,} / {G.number_of_edges():,}")
print(f"Edges missing length_norm: {missing_length_norm:,} / {G.number_of_edges():,}")

slope_norms = [x for x in slope_norms if x is not None]
length_norms = [x for x in length_norms if x is not None]
slope_scores = [x for x in slope_scores if x is not None]

print(f"\nslope_norm  → min={min(slope_norms):.4f}, max={max(slope_norms):.4f}, mean={np.mean(slope_norms):.4f}")
print(f"length_norm → min={min(length_norms):.4f}, max={max(length_norms):.4f}, mean={np.mean(length_norms):.4f}")
print(f"slope_score → min={min(slope_scores):.2f}%, max={max(slope_scores):.2f}%, mean={np.mean(slope_scores):.2f}%")

# Show a steep sample edge so you can verify the normalization manually
print("\nSample steep edges (slope_norm > 0.5):")
count = 0
for u, v, d in G.edges(data=True):
    if d.get("slope_norm", 0) > 0.5 and count < 3:
        sn = d.get("slope_norm")
        ss = d.get("slope_score")
        ln = d.get("length_norm")
        lm = d.get("length_m")
        print(f"  slope_score={ss:.2f}%  slope_norm={sn:.4f}  |  length_m={lm:.1f}m  length_norm={ln:.4f}")
        count += 1

print("\nExpected: slope_norm and length_norm both in [0.0, 1.0]")
print("If slope_norm max < 0.5 or length_norm max is in the hundreds, the graph cache is stale.")
