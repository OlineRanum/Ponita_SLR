import matplotlib.pyplot as plt
import numpy as np

def plot_pose_frame(xy_coords, edges, frame_idx=0, save_path="pose_plot.png"):
    """
    Plot a single frame of pose data showing nodes and edges.
    
    Args:
        xy_coords: Array with shape [time, nodes, 2] containing x,y coordinates
        edges: List of edge connections [[node1, node2], ...]
        frame_idx: Index of frame to plot (default: 0)
        save_path: Path to save the plot
    """
    # Extract the specific frame
    frame_data = xy_coords[frame_idx]  # [nodes, 2]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot edges
    for edge in edges:
        node1, node2 = edge
        if node1 < len(frame_data) and node2 < len(frame_data):
            x1, y1 = frame_data[node1]
            x2, y2 = frame_data[node2]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=0.6)
    
    # Plot nodes
    x_coords = frame_data[:, 0]
    y_coords = frame_data[:, 1]
    ax.scatter(x_coords, y_coords, s=30, c='red', alpha=0.8)
    
    # Add node index labels
    for i, (x, y) in enumerate(frame_data):
        ax.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points', 
                   fontsize=8, alpha=0.9, color='black', fontweight='bold')
    
    # Set properties
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'Pose Frame {frame_idx} - {len(frame_data)} nodes, {len(edges)} edges')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Simple test
    print("Testing plot function...")
    n_frames, n_nodes = 5, 10
    dummy_coords = np.random.rand(n_frames, n_nodes, 2) * 100
    dummy_edges = [[i, i+1] for i in range(n_nodes-1)]
    plot_pose_frame(dummy_coords, dummy_edges, save_path="test_pose.png")
