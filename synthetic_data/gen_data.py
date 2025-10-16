#!/usr/bin/env python3
from data_loader import DataLoader
from plot_sample_data import plot_pose_frame, plot_pose_sequence_movie
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

    # Transform pose and plot
    transformer = TransformPose(base_pose=sample['xy_coords'], time_index=0)
    reset_hand_pose = transformer.reset_hand_to_midpoint()
    plot_pose_frame(reset_hand_pose, sample['edges'], frame_idx=0, save_path="pose_visualizations/pose_reset_hands.png")
    
    # Generate synthetic motion and create animations
    print("\nGenerating synthetic motion...")
    input_pose = sample['xy_coords'][:1]  # Shape (1, 116, 2)
    
    plot = False
    if plot:
        # Generate motion along x-axis
        synthetic_x = transformer.generate_synthetic_motion(
            pose=input_pose, 
            axis=0,  # x-axis motion
            T=40,    # 40 frames
            delta_percent=0.06
        )
        
        # Generate motion along y-axis
        synthetic_y = transformer.generate_synthetic_motion(
            pose=input_pose, 
            axis=1,  # y-axis motion
            T=35,    # 35 frames
            delta_percent=0.08
        )
        
        print("\nCreating animation movies...")
        
        # Create animations
        plot_pose_sequence_movie(
            synthetic_x, 
            sample['edges'], 
            save_path="pose_visualizations/synthetic_x_motion.gif",
            interval=150
        )
        
        plot_pose_sequence_movie(
            synthetic_y, 
            sample['edges'], 
            save_path="pose_visualizations/synthetic_y_motion.gif",
            interval=180
        )
        
        print("\nPose analysis and animation generation complete!")

    

if __name__ == "__main__":
    main()
