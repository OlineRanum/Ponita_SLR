import numpy as np 

class TransformPose():
    def __init__(self, base_pose, time_index = 0):
        self.base_pose = base_pose[0]
        print(f"Base pose shape: {self.base_pose.shape}")
        self.right_wrist = 95
        self.right_hand = np.arange(96, 116)  # Only actual right hand nodes (96-115)
        self.left_wrist = 74
        self.left_hand = np.arange(75, 95)    # Only actual left hand nodes (75-94)
        self.right_shoulder = 1
        self.left_shoulder = 0
        self.right_hip = 5
        self.left_hip = 4

        self.midpoint = self.get_mid_shoulder_hip()
        print(f"Midpoint: {self.midpoint}")

    def __call__(self, kps):
        # kps: [n_frames, 75 nodes, 3 (x,y,confidence)]
        kps = np.array(kps)
        if self.remove_root:
            # Remove root (first keypoint)
            kps = kps[:, 1:, :]
        return kps
    
    def get_mid_shoulder_hip(self):
        right_mid = (self.base_pose[self.right_shoulder] + self.base_pose[self.right_hip]) / 2
        left_mid = (self.base_pose[self.left_shoulder] + self.base_pose[self.left_hip]) / 2
        return (right_mid + left_mid) / 2

    def reset_hand_to_midpoint(self):
        new_base_pose = self.base_pose.copy()
        
        # Calculate midpoint between right shoulder (1) and right hip (5) 
        right_midpoint = (self.base_pose[self.right_shoulder] + self.base_pose[self.right_hip]) / 2
        print(f"Right midpoint (nodes 1-5): {right_midpoint}")
        
        # Calculate midpoint between left shoulder (0) and left hip (4)
        left_midpoint = (self.base_pose[self.left_shoulder] + self.base_pose[self.left_hip]) / 2
        print(f"Left midpoint (nodes 0-4): {left_midpoint}")
        
        # Calculate offset needed to move right wrist to right midpoint
        right_wrist_current = self.base_pose[self.right_wrist]
        right_offset = right_midpoint - right_wrist_current
        print(f"Right offset: {right_offset}")
        
        # Calculate offset needed to move left wrist to left midpoint  
        left_wrist_current = self.base_pose[self.left_wrist]
        left_offset = left_midpoint - left_wrist_current
        print(f"Left offset: {left_offset}")
        
        # Shift all right hand nodes (including wrist) by the same offset
        new_base_pose[self.right_hand] = self.base_pose[self.right_hand] + right_offset
        new_base_pose[self.right_wrist] = self.base_pose[self.right_wrist] + right_offset
        
        # Shift all left hand nodes (including wrist) by the same offset
        new_base_pose[self.left_hand] = self.base_pose[self.left_hand] + left_offset
        new_base_pose[self.left_wrist] = self.base_pose[self.left_wrist] + left_offset
        
        print(f"Shifted {len(self.right_hand)} right hand nodes by offset")
        print(f"Shifted {len(self.left_hand)} left hand nodes by offset")
        
        return new_base_pose[np.newaxis, :, :]
    
    def generate_synthetic_motion(self, pose, axis=0, T=100, delta_percent=0.05, output_original_size=False):
        """
        Generate synthetic data with random motion along specified axis.
        
        Args:
            pose: Input pose shape (1, N, 2) where N is number of nodes
            axis: 0 for x-axis motion, 1 for y-axis motion
            T: Number of time steps to generate
            delta_percent: Step size as percentage of shoulder width (default 5%)
            output_original_size: If True, output shape (T, 133, 2), else (T, N, 2)
            
        Returns:
            numpy.ndarray: Synthetic motion data with shape (T, N, 2) or (T, 133, 2)
        """
        # Get input dimensions
        _, N, _ = pose.shape
        
        # Calculate delta based on shoulder width
        left_shoulder_pos = pose[0, self.left_shoulder]
        right_shoulder_pos = pose[0, self.right_shoulder]
        shoulder_width = abs(right_shoulder_pos[0] - left_shoulder_pos[0])
        delta = delta_percent * shoulder_width
        
        print(f"Shoulder width: {shoulder_width:.4f}")
        print(f"Delta step size: {delta:.4f} ({delta_percent*100}% of shoulder width)")
        
        # Determine output size
        output_N = 133 if output_original_size else N
        
        # Initialize output array
        synthetic_data = np.zeros((T, output_N, 2))
        
        if output_original_size:
            # Need to expand back to original 133 nodes
            # This would require reverse mapping from reduced to original indices
            # For now, just place the reduced pose in the first N positions
            for t in range(T):
                synthetic_data[t, :N] = pose[0]
        else:
            # Set initial pose for all time steps
            for t in range(T):
                synthetic_data[t] = pose[0]
        
        # Generate random motion for hand nodes
        all_hand_nodes = np.concatenate([self.left_hand, [self.left_wrist], 
                                       self.right_hand, [self.right_wrist]])
        
        for t in range(1, T):
            # Copy previous frame
            synthetic_data[t] = synthetic_data[t-1].copy()
            
            # Generate one random step for each hand (left and right move together as units)
            left_hand_nodes = np.concatenate([self.left_hand, [self.left_wrist]])
            right_hand_nodes = np.concatenate([self.right_hand, [self.right_wrist]])
            
            # Random step for left hand: -1, 0, or +1
            left_random_step = np.random.choice([-1, 0, 1])
            left_motion = left_random_step * delta
            
            # Random step for right hand: -1, 0, or +1  
            right_random_step = np.random.choice([-1, 0, 1])
            right_motion = right_random_step * delta
            
            # Apply same motion to all nodes in each hand
            for node_idx in left_hand_nodes:
                if node_idx < output_N:  # Make sure index is valid
                    synthetic_data[t, node_idx, axis] += left_motion
                    
            for node_idx in right_hand_nodes:
                if node_idx < output_N:  # Make sure index is valid
                    synthetic_data[t, node_idx, axis] += right_motion
        
        print(f"Generated synthetic motion: shape {synthetic_data.shape}")
        print(f"Motion along axis {axis} ({'x' if axis == 0 else 'y'})")
        
        return synthetic_data 