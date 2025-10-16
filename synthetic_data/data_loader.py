import os
import pickle
import numpy as np
from typing import Dict, List, Tuple

class DataLoader:
    """
    DataLoader class for pose data with structure:
    - Dimension 0: time/frames
    - Dimension 1: nodes/keypoints 
    - Dimension 2: features (where indices 0,1 are x,y coordinates)
    """
    
    def __init__(self, test_path=None):
        """
        Initialize DataLoader with path to pose data file.
        
        Args:
            test_path (str): Path to test features pickle file
        """
        self.test_path = test_path or '/home/or0007/gitlab/beyondbleu/out/features/H2S/h2s_test_features.pkl'
        
        # Define edges and keypoints to remove
        self.default_edges = [
            [5, 6], [5, 7], [5, 11], [6, 8], [6, 12], [7, 91], [8, 112], [11, 12],
            [13, 15], [15, 17], [15, 18], [15, 19], [91, 92], [91, 96], [91, 100], 
            [91, 104], [91, 108], [92, 93], [93, 94], [94, 95], [96, 97], [97, 98], 
            [98, 99], [100, 101], [101, 102], [102, 103], [104, 105], [105, 106], 
            [106, 107], [108, 109], [109, 110], [110, 111], [112, 113], [112, 117], 
            [112, 121], [112, 125], [112, 129], [113, 114], [114, 115], [115, 116],
            [117, 118], [118, 119], [119, 120], [121, 122], [122, 123], [123, 124],
            [125, 126], [126, 127], [127, 128], [129, 130], [130, 131], [131, 132]
        ]
        self.points_to_remove = [0, 1, 2, 3, 4, 9, 10, 13, 14, 15, 17, 16, 22, 21, 18, 19, 20]
        
        # Load and process data automatically
        self.pose_data = self.load_pose_data()
        self.reduced_data = self._apply_reduction_to_all()
        
    def load_pose_data(self):
        """
        Load pose data from pickle file.
        
        Returns:
            dict: Pose data from test file
        """
        print('Loading pose data...')
        
        # Load test features
        try:
            with open(self.test_path, 'rb') as file:
                feat_test = pickle.load(file)
            print(f"Successfully loaded test features: {len(feat_test)} entries")
        except FileNotFoundError:
            print("Test features file not found")
            feat_test = {}
        
        # Use test features as pose data
        self.pose_data = feat_test
        print(f"Total pose entries loaded: {len(self.pose_data)}")
        
        return self.pose_data
    
    def _apply_reduction_to_all(self):
        """Apply node reduction to all loaded pose data."""
        reduced_data = {}
        
        # Get mapping from first pose
        first_key = list(self.pose_data.keys())[0]
        first_pose = self.pose_data[first_key]
        _, _, old_to_new, _ = self.reduce_keypoints(first_pose[0].numpy() if hasattr(first_pose, 'numpy') else first_pose[0], 
                                                   self.points_to_remove)
        
        # Reduce edges once
        self.reduced_edges = self.reduce_edges(self.default_edges, old_to_new)
        
        # Apply reduction to all poses
        for key, pose_data in self.pose_data.items():
            reduced_pose, _, _ = self.reduce_pose_sequence(pose_data, self.points_to_remove)
            xy_coords = reduced_pose[:, :, :2]  # Extract x,y coordinates
            reduced_data[key] = {
                'pose_data': reduced_pose,
                'xy_coords': xy_coords
            }
            
        return reduced_data
    
    def get_sample(self, key=None):
        """Get a reduced pose sample with edges."""
        if key is None:
            key = list(self.reduced_data.keys())[0]
        
        return {
            'key': key,
            'xy_coords': self.reduced_data[key]['xy_coords'],
            'edges': self.reduced_edges
        }
    
    def get_available_keys(self, limit=10):
        """
        Get list of available pose entry keys.
        
        Args:
            limit (int): Number of keys to return (default: 10)
            
        Returns:
            list: List of pose entry keys
        """
        if not self.pose_data:
            self.load_pose_data()
        
        keys = list(self.pose_data.keys())
        return keys[:limit] if limit else keys
    
    def get_pose_entry(self, key=None):
        """
        Get a specific pose entry by key, or the first entry if no key provided.
        
        Args:
            key (str): Key of the pose entry to retrieve
            
        Returns:
            tuple: (key, pose_data) where pose_data has shape [time, nodes, features]
        """
        if not self.pose_data:
            self.load_pose_data()
            
        if key is None:
            key = list(self.pose_data.keys())[0]
        
        if key not in self.pose_data:
            raise KeyError(f"Key '{key}' not found in pose data")
            
        return key, self.pose_data[key]
    
    def get_xy_coordinates(self, key=None):
        """
        Extract x,y coordinates from pose data.
        
        Args:
            key (str): Key of the pose entry to retrieve
            
        Returns:
            tuple: (key, xy_coords) where xy_coords has shape [time, nodes, 2] (x,y)
        """
        key, pose_data = self.get_pose_entry(key)
        
        # Extract x,y coordinates (indices 0,1 of dimension 2)
        xy_coords = pose_data[:, :, :2]  # [time, nodes, 2]
        
        return key, xy_coords
    
    def reduce_keypoints(self, frame, points_not_use: List[int]) -> Tuple[np.ndarray, List[int], Dict[int, int], Dict[int, int]]:
        """
        Remove specified keypoints from a single frame and return:
        - reduced_frame: (n_kept, features) with only kept keypoints
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
    
    def reduce_pose_sequence(self, pose_data, points_to_remove=None):
        """
        Reduce pose sequence by removing specified keypoints.
        
        Args:
            pose_data: Tensor with shape [time, nodes, features]
            points_to_remove: List of keypoint indices to remove
            
        Returns:
            tuple: (reduced_pose, old_to_new_mapping, new_to_old_mapping)
        """
        if points_to_remove is None:
            points_to_remove = self.points_to_remove
            
        # Convert to numpy if it's a torch tensor
        if hasattr(pose_data, 'numpy'):
            pose_np = pose_data.numpy()
        else:
            pose_np = pose_data
            
        reduced_frames = []
        old_to_new = None
        new_to_old = None
        
        for frame_idx in range(pose_np.shape[0]):
            frame = pose_np[frame_idx]  # [nodes, features]
            reduced_frame, keep_indices, old_to_new, new_to_old = self.reduce_keypoints(frame, points_to_remove)
            reduced_frames.append(reduced_frame)
        
        # Stack the reduced frames
        reduced_pose = np.stack(reduced_frames)
        
        return reduced_pose, old_to_new, new_to_old
    
    def reduce_edges(self, edges_original: List[List[int]], old_to_new: Dict[int, int],
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


if __name__ == "__main__":
    # Simple test
    loader = DataLoader()
    print(f"Loaded {len(loader.pose_data)} poses")
    sample = loader.get_sample()
    print(f"Sample shape: {sample['xy_coords'].shape}")
    print(f"Number of edges: {len(sample['edges'])}")