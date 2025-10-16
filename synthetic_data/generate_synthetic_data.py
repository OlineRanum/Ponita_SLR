from data_transform import TransformPose
import numpy as np
import torch

def generate_synthetic_dataset(base_pose, labels=["horizontal", "vertical"], n_samples=100):
    """
    Generate a synthetic dataset with random motion along specified axis.
    
    Args:
        base_pose (np.ndarray): Base pose to generate synthetic data from, shape (1, 116, 2)
        labels (list): List of labels for the synthetic data
        n_samples (int): Number of samples to generate per label
    
    Returns:
        dict: Dictionary with keys as filename_ids and values as [pose_tensor, label_int]
              Format: {filename_id: [tensor, label]} where label is 0 for horizontal, 1 for vertical
    """
    transformer = TransformPose(base_pose=base_pose, time_index=0)
    synthetic_dataset = {}
    
    # Create label mapping
    label_map = {"horizontal": 0, "vertical": 1}
    
    for label in labels:
        if label == "horizontal":
            axis = 0  # x-axis motion
            T = 40   # 40 frames
            delta_percent = 0.06
        elif label == "vertical":
            axis = 1  # y-axis motion
            T = 35   # 35 frames
            delta_percent = 0.08
        else:
            raise ValueError(f"Unknown label: {label}. Must be 'horizontal' or 'vertical'")
        
        label_int = label_map[label]
        
        for sample_idx in range(n_samples):
            # Generate synthetic sample
            synthetic_sample = transformer.generate_synthetic_motion(
                pose=base_pose, 
                axis=axis, 
                T=T, 
                delta_percent=delta_percent
            )
            
            # Convert to tensor
            pose_tensor = torch.tensor(synthetic_sample, dtype=torch.float32)
            
            # Create filename key
            filename_id = f"synthetic_{label}_{sample_idx:03d}"
            
            # Store in dataset
            synthetic_dataset[filename_id] = [pose_tensor, label_int]
    
    return synthetic_dataset


def save_synthetic_dataset(dataset, save_path="synthetic_dataset.pkl"):
    """
    Save the synthetic dataset to a pickle file.
    
    Args:
        dataset (dict): Synthetic dataset to save
        save_path (str): Path to save the dataset
    """
    import pickle
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Synthetic dataset saved to: {save_path}")
    print(f"Total samples: {len(dataset)}")
    
    # Print some statistics
    labels = [entry[1] for entry in dataset.values()]
    n_horizontal = sum(1 for label in labels if label == 0)
    n_vertical = sum(1 for label in labels if label == 1)
    
    print(f"Horizontal samples (label 0): {n_horizontal}")
    print(f"Vertical samples (label 1): {n_vertical}")


if __name__ == "__main__":
    # Generate train/dev/test datasets
    from data_loader import DataLoader
    from plot_sample_data import plot_pose_sequence_movie
    import os
    import random
    
    print("Generating synthetic datasets for train/dev/test...")
    
    # Create output directory
    output_dir = "/home/or0007/gitlab/Ponita_SLR/datasets/isr/synthetic"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Load base pose
    loader = DataLoader()
    sample = loader.get_sample()
    base_pose = sample['xy_coords'][:1]  # Shape (1, 116, 2)
    
    # Dataset sizes
    train_samples = 200  # 200 per class = 400 total
    dev_samples = 50     # 50 per class = 100 total  
    test_samples = 50    # 50 per class = 100 total
    
    datasets = {
        "train": train_samples,
        "dev": dev_samples, 
        "test": test_samples
    }
    
    all_generated_data = {}  # Store for visualization selection
    
    for split_name, n_samples in datasets.items():
        print(f"\nGenerating {split_name} dataset ({n_samples} samples per class)...")
        
        # Generate dataset
        dataset = generate_synthetic_dataset(
            base_pose=base_pose,
            labels=["horizontal", "vertical"],
            n_samples=n_samples
        )
        
        # Save dataset
        save_path = os.path.join(output_dir, f"synthetic_{split_name}_dataset.pkl")
        save_synthetic_dataset(dataset, save_path)
        
        # Store for visualization
        all_generated_data[split_name] = dataset
    
    print(f"\nAll datasets saved to: {output_dir}")
    
    # Generate visualization GIFs (2 random samples from each class, each split)
    print("\nGenerating visualization GIFs...")
    
    for split_name, dataset in all_generated_data.items():
        # Get samples by class
        horizontal_keys = [k for k, v in dataset.items() if v[1] == 0]
        vertical_keys = [k for k, v in dataset.items() if v[1] == 1]
        
        # Select 2 random samples from each class
        random.seed(42)  # For reproducible selection
        selected_horizontal = random.sample(horizontal_keys, min(2, len(horizontal_keys)))
        selected_vertical = random.sample(vertical_keys, min(2, len(vertical_keys)))
        
        # Generate GIFs
        for i, key in enumerate(selected_horizontal):
            pose_data = dataset[key][0].numpy()  # Convert tensor to numpy
            gif_path = os.path.join(output_dir, "visualizations", f"{split_name}_horizontal_{i+1}_{key}.gif")
            plot_pose_sequence_movie(pose_data, sample['edges'], gif_path, interval=120)
            
        for i, key in enumerate(selected_vertical):
            pose_data = dataset[key][0].numpy()  # Convert tensor to numpy
            gif_path = os.path.join(output_dir, "visualizations", f"{split_name}_vertical_{i+1}_{key}.gif")
            plot_pose_sequence_movie(pose_data, sample['edges'], gif_path, interval=120)
    
    print(f"\nVisualization GIFs saved to: {os.path.join(output_dir, 'visualizations')}")
    
    # Summary
    total_samples = sum(len(dataset) for dataset in all_generated_data.values())
    print(f"\nDataset generation complete!")
    print(f"Total samples generated: {total_samples}")
    print(f"Train: {len(all_generated_data['train'])} samples")
    print(f"Dev: {len(all_generated_data['dev'])} samples") 
    print(f"Test: {len(all_generated_data['test'])} samples")
    print(f"Visualization GIFs: {len(all_generated_data) * 4} files")