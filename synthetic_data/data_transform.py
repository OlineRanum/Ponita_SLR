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
        
        return new_base_pose