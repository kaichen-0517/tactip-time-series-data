import ast
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
 
# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BASE = "./tactile_data/ur5/tactip-Silvia/surface-z-Silvia/run_1/"
FRAME_IDX  = 0        # which time-series frame to visualize

 
# ──────────────────────────────────────────────
# 1. Load CSV
# ──────────────────────────────────────────────
def load_frame(csv_path: str, frame_idx: int):
    """Load one frame from the CSV and return parsed kps and edges."""
    df = pd.read_csv(csv_path, compression="zip")
    print(f"Total frames: {len(df)}")
 
    row = df.iloc[frame_idx]
    print(f"Frame {frame_idx}: {row['time_series_images']}")
    print(f"  num_kps = {row['num_kps']}")
 
    # Parse keypoint positions — list of [x, y]
    kps = np.array(ast.literal_eval(row["kps"]), dtype=float)  # (N, 2)
 
    # Parse edge_index — [[src...], [dst...]] in COO format
    edges_raw = ast.literal_eval(row["edges"])
    src = np.array(edges_raw[0], dtype=int)
    dst = np.array(edges_raw[1], dtype=int)
 
    # Deduplicate: keep only edges where src < dst (undirected graph)
    mask = src < dst
    src, dst = src[mask], dst[mask]
 
    print(f"  keypoints: {kps.shape[0]}  |  unique edges (undirected): {len(src)}")
    return kps, src, dst, row["time_series_images"]
 
 
# ──────────────────────────────────────────────
# 2. Compute edge distances
# ──────────────────────────────────────────────
def compute_distances(kps: np.ndarray, src: np.ndarray, dst: np.ndarray):
    """
    Compute Euclidean distance for each edge.
    Coordinates are in normalized [0, 1] space.
 
    Returns a DataFrame with columns:
      edge_id, src, dst, x_src, y_src, x_dst, y_dst, distance
    """
    diff     = kps[dst] - kps[src]          # (E, 2)
    dists    = np.linalg.norm(diff, axis=1) # (E,)
 
    records = []
    for i, (s, d, dist) in enumerate(zip(src, dst, dists)):
        records.append({
            "edge_id" : i,
            "src"     : int(s),
            "dst"     : int(d),
            "x_src"   : round(float(kps[s, 0]), 6),
            "y_src"   : round(float(kps[s, 1]), 6),
            "x_dst"   : round(float(kps[d, 0]), 6),
            "y_dst"   : round(float(kps[d, 1]), 6),
            "distance": round(float(dist),       6),
        })
 
    df_dist = pd.DataFrame(records).sort_values("distance", ascending=False)
    return df_dist, dists
 
 
# ──────────────────────────────────────────────
# 3. Visualize
# ──────────────────────────────────────────────
def visualize(kps, src, dst, dists, frame_label, out_path):
    """
    Three-panel figure:
      Left   — graph layout: nodes + edges coloured by distance
      Center — distance distribution histogram
      Right  — top-20 longest edges bar chart
    """
    fig = plt.figure(figsize=(18, 6), facecolor="#0f0f1a")
 
    cmap = cm.plasma
    norm = plt.Normalize(vmin=dists.min(), vmax=dists.max())
 
    # ── Panel 1: Graph layout ─────────────────
    ax1 = fig.add_subplot(1, 3, 1, facecolor="#0f0f1a")
 
    # Build line segments for LineCollection (fast rendering)
    segments = np.stack([kps[src], kps[dst]], axis=1)  # (E, 2, 2)
    lc = LineCollection(
        segments,
        array=dists,
        cmap=cmap,
        norm=norm,
        linewidths=0.6,
        alpha=0.65,
    )
    ax1.add_collection(lc)
 
    # Draw nodes
    ax1.scatter(
        kps[:, 0], kps[:, 1],
        s=18, c="#00e5ff", zorder=5,
        edgecolors="white", linewidths=0.3, alpha=0.9,
    )
 
    # Label a few landmark nodes
    for idx in range(0, len(kps), 12):
        ax1.annotate(
            str(idx), (kps[idx, 0], kps[idx, 1]),
            color="#aaaaaa", fontsize=6,
            textcoords="offset points", xytext=(3, 3),
        )
 
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.invert_yaxis()   # image coordinates: y=0 at top
    ax1.set_title(
        f"Graph Layout\n{frame_label.split('/')[-1]}",
        color="white", fontsize=10, fontweight="bold", pad=8,
    )
    ax1.tick_params(colors="#666666", labelsize=7)
    ax1.set_xlabel("x (normalized)", color="#888888", fontsize=8)
    ax1.set_ylabel("y (normalized)", color="#888888", fontsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")
 
    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax1, shrink=0.7, pad=0.02)
    cb.set_label("Edge Distance", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
 
    # ── Panel 2: Distance histogram ───────────
    ax2 = fig.add_subplot(1, 3, 2, facecolor="#0f0f1a")
 
    n_bins = 40
    counts, bin_edges = np.histogram(dists, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_colors  = [cmap(norm(c)) for c in bin_centers]
 
    ax2.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0])*0.9,
            color=bar_colors, edgecolor="none", alpha=0.85)
    ax2.axvline(dists.mean(),  color="#ffffff", linewidth=1.2, linestyle="--",
                label=f"mean = {dists.mean():.4f}")
    ax2.axvline(dists.median() if hasattr(dists, 'median') else float(np.median(dists)),
                color="#00e5ff", linewidth=1.0, linestyle=":",
                label=f"median = {np.median(dists):.4f}")
    ax2.set_title("Edge Distance Distribution", color="white",
                  fontsize=10, fontweight="bold", pad=8)
    ax2.set_xlabel("Distance", color="#888888", fontsize=8)
    ax2.set_ylabel("Count",    color="#888888", fontsize=8)
    ax2.tick_params(colors="#666666", labelsize=7)
    ax2.legend(fontsize=7, framealpha=0.2, labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")
 
    # ── Panel 3: Top-20 longest edges ─────────
    ax3 = fig.add_subplot(1, 3, 3, facecolor="#0f0f1a")
 
    top_n  = 20
    order  = np.argsort(dists)[::-1][:top_n]
    labels = [f"{src[i]}→{dst[i]}" for i in order]
    vals   = dists[order]
    colors = [cmap(norm(v)) for v in vals]
 
    bars = ax3.barh(labels, vals, color=colors, edgecolor="none", alpha=0.85)
    for bar, val in zip(bars, vals):
        ax3.text(val + dists.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", color="#cccccc", fontsize=6.5)
 
    ax3.invert_yaxis()
    ax3.set_title(f"Top {top_n} Longest Edges", color="white",
                  fontsize=10, fontweight="bold", pad=8)
    ax3.set_xlabel("Distance", color="#888888", fontsize=8)
    ax3.tick_params(colors="#666666", labelsize=7)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333333")
 
    # ── Global title ──────────────────────────
    fig.suptitle(
        f"Markers Graph Analysis  |  Frame {FRAME_IDX}  |  "
        f"{len(kps)} nodes · {len(src)} edges",
        color="white", fontsize=12, fontweight="bold", y=1.01,
    )
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print(f"Figure saved → {out_path}")
 
 
# ──────────────────────────────────────────────
# 4. Print summary
# ──────────────────────────────────────────────
def print_summary(df_dist):
    dists = df_dist["distance"]
    print("\n" + "=" * 60)
    print(f"  Total edges  : {len(df_dist)}")
    print(f"  Min distance : {dists.min():.6f}")
    print(f"  Max distance : {dists.max():.6f}")
    print(f"  Mean distance: {dists.mean():.6f}")
    print(f"  Std deviation: {dists.std():.6f}")
    print("=" * 60)
    print("\nTop 10 longest edges:")
    print(df_dist.head(10)[["src","dst","distance"]].to_string(index=False))
    print("\nTop 10 shortest edges:")
    print(df_dist.tail(10)[["src","dst","distance"]].to_string(index=False))
 
 
# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────
def main():
 
    csv_path  = f"{BASE}/markers.zip"
    # Load
    kps, src, dst, frame_label = load_frame(csv_path, FRAME_IDX)
 
    # Compute distances
    df_dist, dists = compute_distances(kps, src, dst)
 
    # Print summary
    print_summary(df_dist)
 
    # Save distance table
    df_dist.to_csv(f"{BASE}/edge_distances.csv", index=False) 
    # Visualize
    visualize(kps, src, dst, dists, frame_label, f"{BASE}/markers_analysis.png")
 
 
if __name__ == "__main__":
    main()