import json
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .pose_transforms_new import CenterAndScaleNormalize

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
        with open(file_path, 'r') as file:
            metadata = json.load(file)
            
        self.gloss_dict = {}
        for item in metadata:  
            self.gloss_dict.setdefault(item['gloss'], []).extend(
                [(instance['video_id'], instance['split'], instance['camera_view']) for instance in item['instances']])


    def _load_pose_data(self, pickle_path):
        """ Load the pickle files and create a dictionary with the data
        """
        labels = {word: index for index, word in enumerate(self.gloss_dict.keys())}

        data_dict = {
            vid_id: {
                'label': labels[gloss],
                'gloss': gloss,
                'node_pos': self._transform_data(pickle.load(open(os.path.join(pickle_path, f'{vid_id}.pkl'), 'rb'))["keypoints"][:, :, :2]),
                'split': split,
                'view': view,
            }
            for gloss, metadata in self.gloss_dict.items()
            for vid_id, split, view in metadata
            if os.path.exists(os.path.join(pickle_path, f'{vid_id}.pkl'))
        }

        return data_dict

    
    #--------------------------------------
    # B. Pre-processing functionalities
    #--------------------------------------

    def _transform_data(self, kps):
        """ Apply selected transformations to the data
        """
        # frames: [2 (x and y), n_frames, 75 nodes]
        frames = torch.tensor(np.asarray(kps, dtype=np.float32)).permute(2, 0, 1)
        
        
        # Subsample nodes
        # frames: [2 (x and y), n_frames, 25 nodes]
        frames = self.pose_select(frames)

        # Downsample number of frames
        if self.downsample:
            frames = self.downsample_frames(frames)

        # Normalize poses
        # TODO Finish testing Scale and Normalization

        if self.set_scalenorm:
            self.scalenorm = CenterAndScaleNormalize()
            frames = self.scalenorm(frames)
            

        
        return frames
    
    
    def downsample_frames(self, frames, downsample_rate = 3):
        return frames[:, ::downsample_rate, :]

    def pose_select(self, frames):
        """ Downsample pose graph based on the standard node selection from holistic 27 minimal node set
        """
        # Indexes for reduction of graph nodes of graph size 27 nodes, predefined in holistic mediapipe package 
        pose_indexes = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]
        return frames[:, :, pose_indexes]

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
        
        graph_constructor = SpatioTemporalGraphBuilder(data_dict, self.args)

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
                'n_frames': data['node_pos'].shape[1], 
                'node_pos': pos,  
                'edges': spatial_edges,   
                'split': data['split'],
                'view': data['view'],
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
    def __init__(self, data_dict, args, inward_edges = None):
        """
        Initialize the graph builder with a fixed number of nodes and a list of inward edges.
        :param num_nodes: Number of nodes in each frame.
        :param inward_edges: List of edges in the format [source, destination].
        """
        # Find max number of frames in dataset
        self.max_n_frames     = max(item['node_pos'].shape[1] for item in data_dict.values())
        self.args             = args
        
        self.N_NODES          = args.n_nodes
        self.tot_number_nodes = self.max_n_frames * self.N_NODES

        if inward_edges is None:
            ## Default holistic mediapipe edges
            self.inward_edges = [ [2, 0], [1, 0], [0, 3], [0, 4], [3, 5], [4, 6], [5, 7], [6, 17], 
                                [7, 8], [7, 9], [9, 10], [7, 11], [11, 12], [7, 13], [13, 14], 
                                [7, 15], [15, 16], [17, 18], [17, 19], [19, 20], [17, 21], [21, 22], 
                                [17, 23], [23, 24], [17, 25], [25, 26]]
            
        else:
            self.inward_edges = inward_edges

        self.n_spatial_edges = len(self.inward_edges)
        self.n_temporal_edges = self.N_NODES

        # Build spatial and temporal edges
        self._build_spatiotemporal_edges()
        # Build node features    
        self._build_node_features()
            
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
            self._load_data(test_data[0], shuffle = False, split='test'),
            self._load_data(test_data[0], shuffle = False, split='test'),
            self._load_data(test_data[0], shuffle = False, split='test'),
        ]

    def _split_dataset(self, data_dict):
        train_data = {k: v for k, v in data_dict.items() if v['split'] == 'train'}
        val_data = {k: v for k, v in data_dict.items() if v['split'] == 'val'}
        test_data_v0 = {k: v for k, v in data_dict.items() if v['split'] == 'test' and v['view'] == 0}
        test_data_v1 = {k: v for k, v in data_dict.items() if v['split'] == 'test' and v['view'] == 1}
        test_data_v2 = {k: v for k, v in data_dict.items() if v['split'] == 'test' and v['view'] == 2}
        test_data = [test_data_v0, test_data_v1, test_data_v2]
        return train_data, val_data, test_data

    def _load_data(self, data_dict, shuffle = True, split = 'train'): 
        data_list = []
        for id, data in data_dict.items():
            pos = data['node_pos'].T
            y = data['label']
            x = data['x']
            if self.args.temporal_configuration == 'spatio_temporal':
                self.edge_index = data['edges']
            
            data_list.append(Data(pos = pos, x = x, edge_index= self.edge_index, y=y, n_frames = data['n_frames'], view = data['view']))
           
        
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












