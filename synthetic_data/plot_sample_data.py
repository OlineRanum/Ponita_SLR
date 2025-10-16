import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def plot_pose_sequence_movie(xy_coords, edges, save_path="pose_animation.gif", fps=10, interval=100):
    """
    Create an animated gif/movie from a temporal pose sequence.
    
    Args:
        xy_coords: Array with shape [time, nodes, 2] containing x,y coordinates
        edges: List of edge connections [[node1, node2], ...]
        save_path: Path to save the animation (supports .gif, .mp4)
        fps: Frames per second for mp4 (default: 10)
        interval: Milliseconds between frames for gif (default: 100)
    """
    n_frames, n_nodes, _ = xy_coords.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Set up the plot limits based on all frames
    x_min, x_max = np.min(xy_coords[:, :, 0]), np.max(xy_coords[:, :, 0])
    y_min, y_max = np.min(xy_coords[:, :, 1]), np.max(xy_coords[:, :, 1])
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # Initialize empty plots
    edge_lines = []
    for _ in edges:
        line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.6)
        edge_lines.append(line)
    
    nodes_scatter = ax.scatter([], [], s=30, c='red', alpha=0.8)
    title_text = ax.set_title('')
    
    # Store node annotations
    node_annotations = []
    for i in range(n_nodes):
        annotation = ax.annotate('', (0, 0), xytext=(3, 3), textcoords='offset points',
                               fontsize=8, alpha=0.9, color='black', fontweight='bold')
        node_annotations.append(annotation)
    
    def animate(frame_idx):
        """Animation function called for each frame."""
        frame_data = xy_coords[frame_idx]
        
        # Update edges
        for i, edge in enumerate(edges):
            node1, node2 = edge
            if node1 < len(frame_data) and node2 < len(frame_data):
                x1, y1 = frame_data[node1]
                x2, y2 = frame_data[node2]
                edge_lines[i].set_data([x1, x2], [y1, y2])
            else:
                edge_lines[i].set_data([], [])
        
        # Update nodes
        x_coords = frame_data[:, 0]
        y_coords = frame_data[:, 1]
        nodes_scatter.set_offsets(np.column_stack([x_coords, y_coords]))
        
        # Update node annotations
        for i, (x, y) in enumerate(frame_data):
            node_annotations[i].set_position((x, y))
            node_annotations[i].set_text(str(i))
        
        # Update title
        title_text.set_text(f'Pose Animation - Frame {frame_idx}/{n_frames-1} - {n_nodes} nodes, {len(edges)} edges')
        
        return edge_lines + [nodes_scatter, title_text] + node_annotations
    
    # Create animation
    print(f"Creating animation with {n_frames} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                 interval=interval, blit=True, repeat=True)
    
    # Save animation
    try:
        if save_path.endswith('.gif'):
            print(f"Saving as GIF: {save_path}")
            anim.save(save_path, writer='pillow', fps=1000/interval)
        elif save_path.endswith('.mp4'):
            print(f"Saving as MP4: {save_path}")
            anim.save(save_path, writer='ffmpeg', fps=fps)
        else:
            # Default to gif
            save_path_gif = save_path + '.gif'
            print(f"No extension specified, saving as GIF: {save_path_gif}")
            anim.save(save_path_gif, writer='pillow', fps=1000/interval)
            save_path = save_path_gif
        
        print(f"Animation saved to: {save_path}")
        
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Try installing required packages: pip install pillow")
        
    plt.close()
    return save_path


if __name__ == "__main__":
    # Simple test
    print("Testing plot functions...")
    n_frames, n_nodes = 10, 20
    
    # Create dummy temporal data with some motion
    dummy_coords = np.random.rand(n_frames, n_nodes, 2) * 100
    
    # Add some smooth motion to make animation interesting
    for t in range(1, n_frames):
        dummy_coords[t] = dummy_coords[t-1] + np.random.randn(n_nodes, 2) * 2
    
    dummy_edges = [[i, i+1] for i in range(n_nodes-1)]
    
    # Test single frame plot
    plot_pose_frame(dummy_coords, dummy_edges, save_path="test_pose.png")
    
    # Test animation
    plot_pose_sequence_movie(dummy_coords, dummy_edges, save_path="test_animation.gif", interval=150)
