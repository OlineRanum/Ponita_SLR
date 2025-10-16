#!/usr/bin/env python3
from data_loader import DataLoader
from plot_sample_data import plot_pose_frame
from data_transform import TransformPose
import numpy as np
def main():
    print("Starting pose analysis...")
    
    # Load data and get a sample
    loader = DataLoader()
    sample = loader.get_sample()
    
    # Print basic info
    print(f"Sample key: {sample['key']}")
    print(f"Shape: {sample['xy_coords'].shape}")
    print(f"Number of edges: {len(sample['edges'])}")
    
    # Plot single frame with node indices
    plot_pose_frame(sample['xy_coords'], sample['edges'], frame_idx=0, save_path="pose_visualizations/pose_with_indices.png")

    print("Analysis complete!")

    transformer = TransformPose(base_pose=sample['xy_coords'], time_index=0)

    reset_hand_pose = transformer.reset_hand_to_midpoint()
    plot_pose_frame(reset_hand_pose[np.newaxis, :, :], sample['edges'], frame_idx=0, save_path="pose_visualizations/pose_reset_hands.png")

if __name__ == "__main__":
    main()
