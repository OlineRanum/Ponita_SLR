import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def reduce_keypoints(frame: np.ndarray, points_not_use: List[int]) -> Tuple[np.ndarray, List[int], Dict[int, int], Dict[int, int]]:
    """
    Remove specified keypoints from a single frame and return:
      - reduced_frame: (n_kept, 2) with only kept keypoints
      - keep_indices: list mapping new_idx -> old_idx
      - old_to_new: dict mapping old_idx -> new_idx
      - new_to_old: dict mapping new_idx -> old_idx
    """
    n_nodes = frame.shape[0]
    points_not_use = sorted(set(i for i in points_not_use if 0 <= i < n_nodes))
    all_indices = set(range(n_nodes))
    keep_indices = sorted(all_indices - set(points_not_use))

    reduced_frame = frame[keep_indices, :]
    old_to_new = {old: new for new, old in enumerate(keep_indices)}
    new_to_old = {new: old for old, new in old_to_new.items()}

    return reduced_frame, keep_indices, old_to_new, new_to_old


def reduce_edges(edges_original: List[List[int]],
                 old_to_new: Dict[int, int],
                 sort_and_dedupe: bool = True) -> List[List[int]]:
    """
    Remap edges from original indices to reduced indices using old_to_new.
    Drops any edge that references a removed node.
    Optionally sorts each pair low→high and dedupes + lexicographically sorts the list.
    """
    remapped: List[List[int]] = []
    for i, j in edges_original:
        if i in old_to_new and j in old_to_new:
            a, b = old_to_new[i], old_to_new[j]
            if a != b:  # avoid self-loops
                if sort_and_dedupe and a > b:
                    a, b = b, a
                remapped.append([a, b])

    if sort_and_dedupe:
        remapped = sorted(set(tuple(e) for e in remapped))
        remapped = [list(e) for e in remapped]

    return remapped


# --- Load your features exactly as before ---
with open('/home/or0007/gitlab/beyondbleu/out/features/H2S/h2s_train_features.pkl', 'rb') as file:
    feat_train = pickle.load(file)

print(feat_train['fz6XzPxdo-0_3-5-rgb_front'].shape)
frame_1 = feat_train['fz6XzPxdo-0_3-5-rgb_front'][0, :, 0:2]  # (N_nodes, 2)

# 1) Define the points to remove (original indices)
points_not_use = [0, 1, 2, 3, 4, 9, 10, 13, 14, 15, 17, 16, 22, 21, 18, 19, 20]
points_not_use = sorted(set(points_not_use))

# 2) Your current corrected edge set (ORIGINAL indices!)
edges_original = [
    [5, 6], [5, 7], [5, 11],
    [6, 8], [6, 12],
    [7, 91],
    [8, 112],
    [11, 12],
    [13, 15],
    [15, 17], [15, 18], [15, 19],
    [91, 92], [91, 96], [91, 100], [91, 104], [91, 108],
    [92, 93], [93, 94], [94, 95],
    [96, 97], [97, 98], [98, 99],
    [100, 101], [101, 102], [102, 103],
    [104, 105], [105, 106], [106, 107],
    [108, 109], [109, 110], [110, 111],
    [112, 113], [112, 117], [112, 121], [112, 125], [112, 129],
    [113, 114], [114, 115], [115, 116],
    [117, 118], [118, 119], [119, 120],
    [121, 122], [122, 123], [123, 124],
    [125, 126], [126, 127], [127, 128],
    [129, 130], [130, 131], [131, 132],
    # Face
    [50,51], [51,52], [52,53], [53,54], [54,55], [55,56], [56,57], [57,58],  # Nose
    [59,60], [60,61], [61,62], [62,63], [63,64], [64,59], [59, 23],          # Right eye + jaw link
    [23, 24], [24, 25], [25, 26], [26, 27], [27, 28],                         # Jaw
    [28, 29], [29, 30], [30, 31], [31, 32], [32, 33],                         # Jaw
    [33, 34], [34, 35], [35, 36], [36, 37], [37, 38], [38, 39],               # Jaw
    [40, 41], [41, 42], [42, 43], [43, 44],                                   # Right brow
    [45, 46], [46, 47], [47, 48], [48, 49],                                   # Left brow
    [65,66], [66,67], [67,68], [68,69], [69,70], [70,65],                     # Left eye
    [71,72], [72,73], [73,74], [74,75], [75,76], [76,77], [77,78], [78,79],
    [79,80], [80,81], [81,82], [82,71],                                       # Outer mouth
    [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], [90,83],   # Inner mouth
    [49, 68], [40, 59], [50, 65], [50, 62], [39, 68], [39, 78], [23, 72],
    [54, 75], [75, 86], [80, 31],                                             # Other face connections
    # Global connections to face (from shoulders)
    [6, 31], [5, 31]
]

# --- Reduce keypoints & edges ---
reduced_frame, keep_indices, old_to_new, new_to_old = reduce_keypoints(frame_1, points_not_use)
updated_edges = reduce_edges(edges_original, old_to_new, sort_and_dedupe=True)

print(f"Original nodes: {frame_1.shape[0]}, kept: {len(keep_indices)}, removed: {len(points_not_use)}")
print("First 10 old->new:", list(old_to_new.items())[:10])
print(f"Edges: original={len(edges_original)}, after_remap={len(updated_edges)}")
# If you want to see the list:
# print(updated_edges)

# --- Plot reduced nodes + updated edges (new indices) ---
fig, ax = plt.subplots(figsize=(8, 8))
x = reduced_frame[:, 0]
y = reduced_frame[:, 1]
valid = ~(np.isnan(x) | np.isnan(y))
ax.scatter(x[valid], y[valid], s=10, zorder=3)

for a, b in updated_edges:
    xi, yi = reduced_frame[a]
    xj, yj = reduced_frame[b]
    if (not np.isnan(xi)) and (not np.isnan(yi)) and (not np.isnan(xj)) and (not np.isnan(yj)):
        ax.plot([xi, xj], [yi, yj], linewidth=1, alpha=0.8, zorder=2)

# Label nodes with NEW indices
for new_idx, (xi, yi) in enumerate(reduced_frame):
    if np.isnan(xi) or np.isnan(yi):
        continue
    ax.annotate(str(new_idx), (xi, yi), xytext=(2, 2), textcoords='offset points', fontsize=6)

ax.invert_yaxis()
ax.set_aspect('equal', adjustable='box')
ax.set_title('Reduced nodes + remapped edges (frame 0)')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('other/test.png', dpi=200)
print("Saved to other/test.png")

# Keep these for later if needed:
#  - keep_indices (new_idx -> old_idx)
#  - old_to_new (old_idx -> new_idx)
