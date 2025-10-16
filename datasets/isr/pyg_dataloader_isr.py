import json
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .pose_transforms_new import CenterAndScaleNormalize
from typing import Dict, List, Tuple

class ISRDataReader:
    def __init__(self, data_dir, args):
        print('Reading data...')
        
        self.args = args
        self.N_NODES = args.n_nodes
        self.set_scalenorm = args.scale_norm
        self.downsample = args.downsample
        # Currently set at max of NGT200 dataset
        self.max_frames = 300
        
        # Load metadata
        file_path = os.path.join(data_dir, args.root_metadata)
        self._load_metadata(file_path)
        
        # Load pose data from pickle files 
        pickle_path = os.path.join(data_dir, args.root_poses)
        data_dict = self._load_pose_data(pickle_path)

        # Define transformations

        
        # Build spatio-temporal graph 
        print('Building graphs...')
        self.data_dict = self._build_spatio_temporal_graph(data_dict)

    def _load_metadata(self, file_path):
        """ Load the metadata from the json file
        """
        import pickle 
        with open(file_path + 'train.h2s_pkl', 'rb') as file:
            metadata_train = pickle.load(file)
        with open(file_path + 'dev.h2s_pkl', 'rb') as file:
            metadata_val = pickle.load(file)
        with open(file_path + 'test.h2s_pkl', 'rb') as file:
            metadata_test = pickle.load(file)
            
        self.gloss_dict = {}
        # TODO: Fix this - currently not loading correctly 
        pos_val = 0
        neg_val = 0
        pos_test = 0
        neg_test = 0
        pos_train = 0
        neg_train = 0
        for key, value in metadata_train.items():  
            if value['sentiment'] != 'neutral':
                self.gloss_dict.setdefault(value['sentiment'], []).extend(
                    [(key, 'train')])
                if value['sentiment'] == 'positive':
                    pos_train += 1
                elif value['sentiment'] == 'negative':
                    neg_train += 1

        for key, value in metadata_val.items():  
            if value['sentiment'] != 'neutral':
                self.gloss_dict.setdefault(value['sentiment'], []).extend(
                    [(key, 'val')])
                if value['sentiment'] == 'positive':
                    pos_val += 1
                elif value['sentiment'] == 'negative':
                    neg_val += 1

        for key, value in metadata_test.items():  
            if value['sentiment'] != 'neutral':
                self.gloss_dict.setdefault(value['sentiment'], []).extend(
                    [(key, 'test')])
                if value['sentiment'] == 'positive':
                    pos_test += 1
                elif value['sentiment'] == 'negative':
                    neg_test += 1
        print(f"Number of positive samples: train {pos_train}, val {pos_val}, test {pos_test}")
        print(f"Number of negative samples: train {neg_train}, val {neg_val}, test {neg_test}")

    def _load_pose_data(self, pickle_path):
        """ Load the pickle files and create a dictionary with the data
        """
        labels = {word: index for index, word in enumerate(self.gloss_dict.keys())}

        with open('/home/or0007/gitlab/beyondbleu/out/features/H2S/h2s_train_features.pkl', 'rb') as file:
            feat_train = pickle.load(file)
        with open('/home/or0007/gitlab/beyondbleu/out/features/H2S/h2s_test_features.pkl', 'rb') as file:
            feat_val = pickle.load(file)
            feat_test = feat_val
        
        features = {**feat_train, **feat_val}
        
        data_dict = {
            vid_id: {
                'label': labels[key],
                'gloss': key,
                'node_pos': self._transform_data(features[vid_id]),
                'split': split
            }
            for key, metadata in self.gloss_dict.items()
            for vid_id, split in metadata
            if key != 'neutral' and vid_id in features
        }
        

        return data_dict

    
    #--------------------------------------
    # B. Pre-processing functionalities
    #--------------------------------------

    def _transform_data(self, kps):
        """ Apply selected transformations to the data
        """
        frames = self.pose_select(kps)

        # frames: [2 (x and y), n_frames, 75 nodes]
        frames = torch.tensor(np.asarray(frames, dtype=np.float32)).permute(2, 0, 1)

        frames = frames[0:2, :, :]  # Only keep x and y coordinates
       

        
        # Subsample nodes
        # frames: [2 (x and y), n_frames, 25 nodes]
        
        # Downsample number of frames
        #if self.downsample:
        #    frames = self.downsample_frames(frames)

        # Normalize poses
        # TODO Finish testing Scale and Normalization

        #if self.set_scalenorm:
        #    self.scalenorm = CenterAndScaleNormalize()
        #     frames = self.scalenorm(frames)

        return frames
    
    
    def downsample_frames(self, frames, downsample_rate = 3):
        return frames[:, ::downsample_rate, :]
    
    def reduce_keypoints(self, frame, points_not_use: List[int]) -> Tuple[np.ndarray, List[int], Dict[int, int], Dict[int, int]]:
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
        self.old_to_new = {old: new for new, old in enumerate(keep_indices)}
        self.new_to_old = {new: old for old, new in self.old_to_new.items()}

        return reduced_frame, keep_indices, self.old_to_new, self.new_to_old

    def pose_select(self, frames):
        """ Downsample pose graph based on the standard node selection from holistic 27 minimal node set
        """
        # Indexes for reduction of graph nodes of graph size 27 nodes, predefined in holistic mediapipe package 
        points_not_use = [0, 1, 2, 3, 4, 9, 10, 13, 14, 15, 17, 16, 22, 21, 18, 19, 20]
        if self.N_NODES == 116:
            points_to_use = np.arange(0,117)
        elif self.N_NODES == 47:
            points_to_use = np.concatenate([np.arange(95, 116), np.arange(74, 95), np.arange(0,5)])
        # Calculate points_not_use as all points from 0-116 that are not in points_to_use
        all_points = set(range(116))
        points_not_use = sorted(list(list(all_points - set(points_to_use)) + points_not_use))

        reduced_frames = []
        for frame in frames:
            reduced_frame, self.keep_indices, old_to_new, new_to_old = self.reduce_keypoints(frame, points_not_use)
            reduced_frames.append(reduced_frame)
        
        # Stack the numpy arrays first, then convert to tensor
        reduced_frames = np.stack(reduced_frames)
        return torch.tensor(reduced_frames, dtype=torch.float32)

    #--------------------------------------
    # C. Graph construction functionalities
    #--------------------------------------    
        
    def _build_spatio_temporal_graph(self, data_dict):
        """
        Builds a spatio-temporal graph from the provided data dictionary.

        Each item in the data dictionary is transformed using the SpatioTemporalGraphBuilder,
        and the resulting spatio-temporal graph data is stored in a new dictionary.

        :param data_dict: A dictionary containing video data.
        :return: A dictionary containing the spatio-temporal graph data.
        """
        
        graph_constructor = SpatioTemporalGraphBuilder(data_dict, self.args, old_to_new=self.old_to_new)

        graph_dict = {}
        max_frames_count = 0
        for vid_id, data in data_dict.items():

            # number of frames per gloss
            n_frames = data['node_pos'].shape[1]
            if n_frames > max_frames_count:
                max_frames_count = n_frames
            end_idx = int(n_frames*self.N_NODES)

            if n_frames < self.max_frames:
                spatial_edges =  graph_constructor.spatial_edges[:int(n_frames*graph_constructor.n_spatial_edges),:]
                spatial_edges = spatial_edges.t().contiguous()
                temporal_edges = graph_constructor.temporal_edges[:int((n_frames-1)*graph_constructor.n_temporal_edges),:]
                temporal_edges = temporal_edges.t().contiguous()

            else: 
                spatial_edges =  graph_constructor.spatial_edges[:int(self.max_frames*graph_constructor.n_spatial_edges),:]
                spatial_edges = spatial_edges.t().contiguous()
                temporal_edges = graph_constructor.temporal_edges[:int((self.max_frames-1)*graph_constructor.n_temporal_edges),:]
                temporal_edges = temporal_edges.t().contiguous()
                


            # Get landmarks as features
            x = graph_constructor.landmark_features[:,:end_idx].T


            # Get positions
            pos = graph_constructor.reshape_nodes(data['node_pos'])

            #x, pos = self.add_padding(x, pos)
            graph_dict[vid_id] = {
                'label': data['label'],
                'gloss': data['gloss'],
                'x': x,  
                'n_frames': data['node_pos'].shape[1], # pos [feat, T, N_nodes]
                'node_pos': pos,  
                'edges': spatial_edges,   
                'split': data['split'],
            }

        return graph_dict

    def add_padding(self, x, pos_data):
        nodes_to_add = self.max_frames*self.N_NODES-pos_data.shape[1]
        if nodes_to_add >= 0:
            pos_data = torch.nn.functional.pad(pos_data, (0, nodes_to_add), "constant", 0)
            x = torch.nn.functional.pad(x, (0, 0, 0, nodes_to_add), "constant", 0)
        elif nodes_to_add < 0:
            pos_data = pos_data[:, :self.max_frames*self.N_NODES]
            x = x[:self.max_frames*self.N_NODES, :]
            
        return x, pos_data
    
class SpatioTemporalGraphBuilder:
    def __init__(self, data_dict, args, inward_edges = None, old_to_new = None):
        """
        Initialize the graph builder with a fixed number of nodes and a list of inward edges.
        :param num_nodes: Number of nodes in each frame.
        :param inward_edges: List of edges in the format [source, destination].
        """
        self.old_to_new = old_to_new    
        # Find max number of frames in dataset
        self.max_n_frames     = max(item['node_pos'].shape[1] for item in data_dict.values())
        self.args             = args
        
        self.N_NODES          = args.n_nodes
        self.tot_number_nodes = self.max_n_frames * self.N_NODES

        if inward_edges is None:
            ## Default holistic mediapipe edges
            self.inward_edges =[
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
            self.inward_edges = self.reduce_edges(self.inward_edges, sort_and_dedupe=True)

            
        else:
            self.inward_edges = inward_edges
            
        self.n_spatial_edges = len(self.inward_edges)
        self.n_temporal_edges = self.N_NODES

        # Build spatial and temporal edges
        self._build_spatiotemporal_edges()
        # Build node features    
        self._build_node_features()

    def reduce_edges(self, edges_original: List[List[int]],
                    sort_and_dedupe: bool = True) -> List[List[int]]:
        """
        Remap edges from original indices to reduced indices using old_to_new.
        Drops any edge that references a removed node.
        Optionally sorts each pair low→high and dedupes + lexicographically sorts the list.
        """
        remapped: List[List[int]] = []
        for i, j in edges_original:
            if i in self.old_to_new and j in self.old_to_new:
                a, b = self.old_to_new[i], self.old_to_new[j]
                if a != b:  # avoid self-loops
                    if sort_and_dedupe and a > b:
                        a, b = b, a
                    remapped.append([a, b])

        if sort_and_dedupe:
            remapped = sorted(set(tuple(e) for e in remapped))
            remapped = [list(e) for e in remapped]

        return remapped
            
    def _build_node_features(self):
        """
        Builds the node features for a given number of frames.
        :param num_frames: The number of frames in the data.
        :return: A tensor with the node features in the graph.
        """
        identity_matrix = np.eye(self.N_NODES)
        landmark_features = np.tile(identity_matrix, (1, self.max_n_frames))
        self.landmark_features = torch.tensor(landmark_features, dtype=torch.float32)

    def _build_spatiotemporal_edges(self):
        """
        Builds the spatio-temporal edges for a given number of frames.
        :param num_frames: The number of frames in the data.
        :return: A tensor representing the edges in the graph.
        """
        spatial_edges = []

        # Adding spatial edges for each frame
        # Indexing incrementally
        for frame in range(self.max_n_frames):

            frame_offset = frame * self.N_NODES
            for edge in self.inward_edges:
                n1 = frame_offset + edge[0]
                n2 = frame_offset + edge[1]
                spatial_edges.append([n1, n2])

        self.spatial_edges = torch.tensor(spatial_edges)

        # Adding temporal edges
        temporal_edges = []
        for frame in range(self.max_n_frames - 1):
            for node in range(self.N_NODES):
                n1 = frame * self.N_NODES + node
                n2 = (frame + 1) * self.N_NODES + node
                temporal_edges.append([n1, n2])

        self.temporal_edges = torch.tensor(temporal_edges)

    
    def reshape_nodes(self, pos_data):
        return pos_data.reshape(pos_data.shape[0], -1)
    

class ISRDataLoader:
    def __init__(self, data, args):
        print('Building dataloader...')
        self.data_dict = data.data_dict
        self.batch_size = args.batch_size
        self.args = args
        
        if args.temporal_configuration == 'per_frame':
            self.inward_edges = [ [2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], 
                                [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], 
                                [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], 
                                [17, 23], [23, 24], [17, 25], [25, 26]]
            self.edge_index = torch.tensor(self.inward_edges, dtype=torch.long).t().contiguous()
        
        self.build_loaders()


    def build_loaders(self):
        train_data, val_data, test_data = self._split_dataset(self.data_dict)
        self.train_loader  = self._load_data(train_data)
        self.val_loader = self._load_data(val_data, shuffle = False, split = 'val')
        self.test_loader = [
            self._load_data(test_data[0], shuffle = False, split='test')
        ]

    def _split_dataset(self, data_dict):
        train_data = {k: v for k, v in data_dict.items() if v['split'] == 'train'}
        val_data = {k: v for k, v in data_dict.items() if v['split'] == 'test'}
        test_data = [{k: v for k, v in data_dict.items() if v['split'] == 'test' }]
        return train_data, val_data, test_data

    def _load_data(self, data_dict, shuffle = True, split = 'train'): 
        data_list = []
        for id, data in data_dict.items():
            pos = data['node_pos'].T
            y = data['label']
            x = data['x']
            if self.args.temporal_configuration == 'spatio_temporal':
                self.edge_index = data['edges']
            
            data_list.append(Data(pos = pos, x = x, edge_index= self.edge_index, y=y, n_frames = data['n_frames'], view = 1))
           
        
        print('Number of ' + split + ' points:', len(data_list))
        
        return DataLoader(data_list, batch_size=self.batch_size, shuffle=shuffle)
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ISR Dataset
    parser.add_argument('--root', type=str, default="",
                        help='Data set location')
    parser.add_argument('--root_metadata', type=str, default="subset_metadata.json",
                        help='Metadata json file location')
    parser.add_argument('--root_poses', type=str, default="subset_selection",
                        help='Pose data dir location')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size. Does not scale with number of gpus.')
    parser.add_argument('--temporal_configuration', type=str, default="spatio_temporal",
                        help='Temporal configuration of the graph. Options: spatio_temporal, per_frame') 
    
    ## Graph size parameter
    parser.add_argument('--n_nodes', type=int, default=27,
                        help='Number of nodes to use when reducing the graph - only 27 currently implemented') 
    # Arg parser
    args = parser.parse_args()

    data_dir = os.path.dirname(__file__) + '/' + args.root
    data = ISRDataReader(data_dir, args)

    pyg_loader = ISRDataLoader(data, args)
    pyg_loader.build_loaders()












