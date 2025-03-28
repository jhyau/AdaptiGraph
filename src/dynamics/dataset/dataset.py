import numpy as np
import torch 
from torch.utils.data import Dataset

from sim.utils import load_yaml
from dynamics.utils import pad, pad_torch
from dynamics.dataset.load import load_dataset, load_positions, lazy_load_positions, get_position_paths
from dynamics.dataset.graph import fps, construct_edges_from_states

class DynDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        material_config,
        phase='train',
        lazy_loading=False,
    ):
        assert phase in ['train', 'valid']
        print(f'Loading {phase} dataset...')
        self.phase = phase
        
        self.dataset_config = dataset_config
        self.material_config = material_config
        self.verbose = dataset_config['verbose']
        
        ## general info
        self.n_his = self.dataset_config['n_his']
        self.n_future = self.dataset_config['n_future']
        self.store_rest_state = self.dataset_config['store_rest_state']
        
        self.add_randomness = self.dataset_config['randomness']['use']
        self.state_noise = self.dataset_config['randomness']['state_noise'][self.phase]
        self.phys_noise = self.dataset_config['randomness']['phys_noise'][self.phase]
        self.lazy_loading = lazy_loading
        
        ## objects info
        self.obj_config = self.dataset_config['datasets']
        assert len(self.obj_config) == 1, "Only one object type is supported."
        self.dataset = self.obj_config[0]
        # node
        self.max_nobj = self.dataset['max_nobj']
        self.fps_radius_range = self.dataset['fps_radius_range']
        # relation
        self.max_nR = self.dataset['max_nR']
        self.adj_radius_range = self.dataset['adj_radius_range']
        self.topk = self.dataset['topk']
        # kNN for tool to object relations
        self.knn_range = self.dataset['knn_range']
        if "min_knn" in self.dataset:
            self.min_kNN = self.dataset['min_knn']
        else:
            self.min_kNN = 1.0
        if "knn_increment" in self.dataset:
            self.knn_increment = self.dataset['knn_increment']
        else:
            self.knn_increment = 0.1
        # type of tool to object connection
        self.connect_tool_all = self.dataset['connect_tool_all']
        self.connect_tool_all_non_fixed = self.dataset['connect_tool_all_non_fixed']
        if "connect_tool_surface" in self.dataset:
            self.connect_tool_surface = self.dataset['connect_tool_surface']
            self.connect_tool_surface_ratio = self.dataset['connect_tool_surface_ratio']
            print(f"connecting tool to surface: {self.connect_tool_surface}, surface ratio: {self.connect_tool_surface_ratio}")
        else:
            self.connect_tool_surface = False
            self.connect_tool_surface_ratio = 1.0
        
        # load data pairs, all object particles and determined by eef delta position
        # pair_list: (T, 8), [episode_idx (0), pair (1:8)]
        # physics_params: (n_epis, phys_dim)
        self.pair_lists, self.physics_params, self.lowest_epi_num = load_dataset(dataset_config, material_config, phase) 
        self.pair_lists = np.array(self.pair_lists)
        print(f'Length of dataset is {self.pair_lists.shape}. \n')
        
        self.materials = {}
        for k in self.physics_params[0].keys():
                self.materials[k] = self.physics_params[0][k].shape[0]
        
        # load positions and physics parameters
        # eef_pos: (n_epis, T, N_eef, 3)
        # obj_pos: (n_epis, T, N_obj, 3)
        if not lazy_loading:
            print(f"not lazy loading: {lazy_loading}. Loading full dataset into memory...")
            self.eef_pos, self.obj_pos = load_positions(dataset_config)
            self.pos_dim = self.obj_pos[0].shape[-1]
            # get dimensions
            self.eef_dim = self.eef_pos[0].shape[1]
        else:
            # Do lazy loading instead, only load in the particle positions when needed instead of all at once
            # eef_pos: (T, N_eef, 3)
            # obj_pos: (T, N_obj, 3)
            print(f"lazy load: {lazy_loading}. Doing lazy loading...")
            self.positions_paths = get_position_paths(dataset_config)
            self.eef_pos_0, self.obj_pos_0 = lazy_load_positions(self.positions_paths, 0)
            self.pos_dim = self.obj_pos_0.shape[-1]
            print("all particle position paths: ", self.positions_paths)
            # get dimensions
            self.eef_dim = self.eef_pos_0.shape[1]

        ## Save the "rest" state of the object at time step 0 for each episode
        # obj_rest_states: (n_epis, N_obj, 3)
        #self.obj_rest_states = self.obj_pos[:, 0, :, :]
        # TODO: For now, to maintain n_his for eef as well. Later see if there's a way to remove this
        # eef_rest_states: (n_epis, N_eef, 3)
        #self.eef_rest_states = self.eef_pos[:, 0, :, :]
        #print(f"object particle rest states size: {self.obj_rest_states.shape}, eef rest state size: {self.eef_rest_states.shape}")
        
        # get dimensions
        self.obj_dim = self.max_nobj
        self.state_dim = self.obj_dim + self.eef_dim
        if self.verbose:
            print(f"object dimension: {self.obj_dim}, eef dimension: {self.eef_dim}.")
        
    def __len__(self):
        return len(self.pair_lists)
    
    def __getitem__(self, idx):
        episode_idx = self.pair_lists[idx][0].astype(int)
        pair = self.pair_lists[idx][1:].astype(int)
        print(f"pair size: {len(pair)}")
        if self.store_rest_state and len(pair) != self.n_his + self.n_future:
            assert len(pair) == self.n_his - 1 + self.n_future
        else:
            assert len(pair) == self.n_his + self.n_future

        if self.lazy_loading:
            # Do lazy loading instead, only load in the particle positions when needed instead of all at once
            # eef_pos: (T, N_eef, 3)
            # obj_pos: (T, N_obj, 3)
            eef_pos, obj_pos = lazy_load_positions(self.positions_paths, episode_idx)

        ## get history keypoints
        obj_kps = []
        eef_kps = []
        # Store the "rest" position at time step 0 first before the history of steps if it wasn't added to the frame pairs
        # in preprocessing
        if self.store_rest_state and len(pair) != self.n_his + self.n_future:
            print(f"*****************Storing the rest state*****************")
            if not self.lazy_loading:
                obj_kps.append(self.obj_pos[episode_idx][0])
                eef_kps.append(self.eef_pos[episode_idx][0])
            else:
                obj_kps.append(obj_pos[0])
                eef_kps.append(eef_pos[0])
        for i in range(len(pair)):
            frame_idx = pair[i]
            # eef keypoints
            if not self.lazy_loading:
                eef_kp = self.eef_pos[episode_idx][frame_idx] # (N_eef, 3)
                # object keypoints
                obj_kp = self.obj_pos[episode_idx][frame_idx] # (N_obj, 3)
            else:
                eef_kp = eef_pos[frame_idx]
                # object keypoints
                obj_kp = obj_pos[frame_idx]
            obj_kps.append(obj_kp)
            eef_kps.append(eef_kp)
        
        # obj_kps: (T, N_obj_all, 3)
        # eef_kps: (T, N_eef, 3)
        obj_kps, eef_kps = np.array(obj_kps), np.array(eef_kps)
        
        ## Farthest point sampling for object particles
        # fps_obj_idx: (N_fps, )
        obj_kp_start = obj_kps[self.n_his-1] # (N_obj, 3)
        fps_idx_list = fps(obj_kp_start, self.max_nobj, self.fps_radius_range, verbose=self.verbose)
        obj_kp_num = len(fps_idx_list)
        
        ## process object keypoints
        # fps_obj_kps: (T, N_obj, 3)
        fps_obj_kp = obj_kps[:, fps_idx_list] # (T, N_fps, 3)
        fps_obj_kps = pad(fps_obj_kp, self.max_nobj, dim=1) # (T, N_obj, 3)
        
        ## get current state delta (action)
        # states_delta: (N_obj + N_eef, 3)
        eef_kp = np.stack(eef_kps[self.n_his-1 : self.n_his+1], axis=0) # (2, N_eef, 3)
        eef_kp_num = eef_kp.shape[1]
        states_delta = np.zeros((self.state_dim, self.pos_dim), dtype=np.float32)
        states_delta[self.obj_dim : self.obj_dim + eef_kp_num] = eef_kp[1] - eef_kp[0]
        if self.verbose:
            print(f"current state delta: {states_delta.shape}.")
        
        ## load history states
        # state_history: (n_his, N_obj + N_eef, 3)
        ## Keep "rest" state (the positions and info at time 0) in state history: (n_his+1, N_obj + N_eef, 3)
        max_y = 0
        min_y = 0
        max_x = 0
        max_z = 0
        min_x = 0
        min_z = 0
        state_history = np.zeros((self.n_his, self.state_dim, self.pos_dim), dtype=np.float32)
        for fi in range(self.n_his):
            obj_kp_his = fps_obj_kps[fi] # (N_obj, 3)
            max_y = np.max(obj_kp_his[:,1])
            min_y = np.min(obj_kp_his[:,1])
            max_x = np.max(obj_kp_his[:,0])
            max_z = np.max(obj_kp_his[:,2])
            min_x = np.min(obj_kp_his[:,0])
            min_z = np.min(obj_kp_his[:,2])
            eef_kp_his = eef_kps[fi] # (N_eef, 3)
            state_history[fi] = np.concatenate([obj_kp_his, eef_kp_his], axis=0)
        if self.verbose:
            print(f"history states: {state_history.shape}.")
        min_x = (max_x - min_x) * (1 - self.connect_tool_surface_ratio) + min_x
        min_z = (max_z - min_z) * (1 - self.connect_tool_surface_ratio) + min_z
        max_y = max_y * self.connect_tool_surface_ratio #0.8
        max_x = max_x * self.connect_tool_surface_ratio
        max_z = max_z * self.connect_tool_surface_ratio

        ## load future states
        # future objects: (n_future, N_obj, 3)
        obj_kp_future = np.zeros((self.n_future, self.obj_dim, self.pos_dim), dtype=np.float32)
        for fi in range(self.n_future):
            obj_kp_fu = fps_obj_kps[self.n_his+fi] # (N_obj, 3)
            obj_kp_future[fi] = obj_kp_fu
        
        # future action: (n_future - 1, N_obj + N_eef, 3)
        # future eef: (n_future - 1, N_obj + N_eef, 3)
        states_delta_future = np.zeros((self.n_future - 1, self.state_dim, self.pos_dim), dtype=np.float32)
        eef_future = np.zeros((self.n_future - 1, self.state_dim, self.pos_dim), dtype=np.float32)
        for fi in range(self.n_future - 1):
            eef_kp_fu = np.stack(eef_kps[self.n_his+fi : self.n_his+fi+2], axis=0) # (2, N_eef, 3)
            eef_future[fi, self.obj_dim : self.obj_dim + eef_kp_num] = eef_kp_fu[0] # (N_eef, 3)
            states_delta_future[fi, self.obj_dim : self.obj_dim + eef_kp_num] = eef_kp_fu[1] - eef_kp_fu[0] # (N_eef, 3)
        
        if self.verbose:
            print(f"future obj states: {obj_kp_future.shape}.")
            print(f"future action: {states_delta_future.shape}; future eef: {eef_future.shape}.")
        
        ## load masks
        # state_mask: (N_obj + N_eef, )
        # eef_mask: (N_obj + N_eef, )
        state_mask = np.zeros((self.state_dim), dtype=bool)
        state_mask[:obj_kp_num] = True
        state_mask[self.max_nobj : self.max_nobj + eef_kp_num] = True
        
        eef_mask = np.zeros((self.state_dim), dtype=bool)
        eef_mask[self.obj_dim : self.obj_dim + eef_kp_num] = True
        
        obj_mask = np.zeros((self.obj_dim), dtype=bool)
        obj_mask[:obj_kp_num] = True
        
        ## construct attrs
        # TODO: change to include more than one object, start with 2 or 3 objects like obj1 + obj2 + eef
        # Node attribute: indicate whether particle belongs to object or eef
        # attr_dim: (N_obj + N_eef, 2)
        attr_dim = 2 # object + eef
        attrs = np.zeros((self.state_dim, attr_dim), dtype=np.float32)
        attrs[:obj_kp_num, 0] = 1.
        attrs[self.max_nobj : self.max_nobj + eef_kp_num, 1] = 1.
        
        ## construct instance information
        # TODO: update instance info here too
        instance_num = 1
        p_rigid = np.zeros(instance_num, dtype=np.float32)
        p_instance = np.zeros((self.max_nobj, instance_num), dtype=np.float32)
        p_instance[:obj_kp_num, 0] = 1

        ## construct physics information
        physics_param = self.physics_params[episode_idx]  # dict
        for material_name in self.dataset_config['materials']:
            if material_name not in physics_param.keys():
                raise ValueError(f'Physics parameter {material_name} not found in {self.dataset_config["data_dir"]}')
            physics_param[material_name] += np.random.uniform(-self.phys_noise, self.phys_noise, 
                    size=physics_param[material_name].shape)
        
        ## construct physics information for each particle
        material_idx = np.zeros((self.max_nobj, len(self.material_config['material_index'])), dtype=np.int32)
        assert len(self.dataset_config['materials']) == 1, 'only support single material'
        material_idx[:obj_kp_num, self.material_config['material_index'][self.dataset_config['materials'][0]]] = 1
            
        ## add randomness
        if self.add_randomness:
            state_history += np.random.uniform(-self.state_noise, self.state_noise, size=state_history.shape)
            # rotation randomness (already translation-invariant)
            random_rot = np.random.uniform(-np.pi, np.pi)
            rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
                                [np.sin(random_rot), np.cos(random_rot), 0],
                                [0, 0, 1]], dtype=state_history.dtype)  # 2D rotation matrix in xy plane
            state_history = state_history @ rot_mat[None]
            states_delta = states_delta @ rot_mat
            eef_future = eef_future @ rot_mat[None]
            states_delta_future = states_delta_future @ rot_mat[None]
            obj_kp_future = obj_kp_future @ rot_mat[None]
        
        ## numpy to torch
        state_history = torch.from_numpy(state_history).float()
        states_delta = torch.from_numpy(states_delta).float()
        eef_future = torch.from_numpy(eef_future).float()
        states_delta_future = torch.from_numpy(states_delta_future).float()
        obj_kp_future = torch.from_numpy(obj_kp_future).float()
        attrs = torch.from_numpy(attrs).float()
        state_mask = torch.from_numpy(state_mask)
        eef_mask = torch.from_numpy(eef_mask)
        obj_mask = torch.from_numpy(obj_mask)
        p_rigid = torch.from_numpy(p_rigid).float()
        p_instance = torch.from_numpy(p_instance).float()
        physics_param = {k: torch.from_numpy(v).float() for k, v in physics_param.items()}
        material_idx = torch.from_numpy(material_idx).long()

        ## construct edges
        # Rr, Rs: (n_rel, N)
        adj_thresh = np.random.uniform(*self.adj_radius_range)
        if self.min_kNN < 1.0:
            knn_thresh = np.random.uniform(*self.knn_range)
        else:
            knn_thresh = 1.0
        print(f"knn_thresh: {knn_thresh}")
        Rr, Rs = construct_edges_from_states(state_history[-1], adj_thresh, state_mask, eef_mask, 
                                             self.topk, self.connect_tool_all, max_y=max_y, min_y=min_y,
                                             max_x=max_x, max_z=max_z,
                                             connect_tools_surface=self.connect_tool_surface,
                                             connect_tool_all_non_fixed=self.connect_tool_all_non_fixed, kNN=knn_thresh)
        # Rr = pad_torch(Rr, self.max_nR)
        # Rs = pad_torch(Rs, self.max_nR)
        exceed_max_nR = True
        kNN = knn_thresh #1.0
        decrease_topK = self.topk
        while(exceed_max_nR):
            # If edge construction exceeds the max allowed edges, first decrease kNN of tool to object connections
            # if that's not sufficient then increase the topK for edge connections
            try:
                Rr = pad_torch(Rr, self.max_nR)
                Rs = pad_torch(Rs, self.max_nR)
                exceed_max_nR = False
            except Exception as e:
                # Exception due to exceeding the max threshold
                exceed_max_nR = True

                # Reduce num of edges in the graph
                if kNN <= self.min_kNN:
                    # We can no longer further reduce the tool to object connections
                    # raise Exception for this case, or start reducing the degrees of edges for each node by increasing topK
                    print(f"!!!!!!!!!!!!!kNN has reached minimum of {self.min_kNN}!!!!!!!")
                    decrease_topK = decrease_topK - 1
                    Rr, Rs = construct_edges_from_states(state_history[-1], adj_thresh, state_mask, eef_mask, 
                                             decrease_topK, self.connect_tool_all, max_y=max_y, min_y=min_y,
                                             max_x=max_x, max_z=max_z,
                                             connect_tools_surface=self.connect_tool_surface,
                                             connect_tool_all_non_fixed=self.connect_tool_all_non_fixed, kNN=kNN)
                else:
                    kNN = kNN - self.knn_increment
                    print(f"reducing kNN to {kNN}")
                    Rr, Rs = construct_edges_from_states(state_history[-1], adj_thresh, state_mask, eef_mask, 
                                                self.topk, self.connect_tool_all, max_y=max_y, min_y=min_y,
                                                max_x=max_x, max_z=max_z,
                                                connect_tools_surface=self.connect_tool_surface,
                                                connect_tool_all_non_fixed=self.connect_tool_all_non_fixed, kNN=kNN)
        
        ## save graph
        graph = {
            # input information
            "state": state_history,  # (n_his, N+M, state_dim)
            "action": states_delta,  # (N+M, state_dim)

            # future information
            "eef_future": eef_future,  # (n_future-1, N+M, state_dim)
            "action_future": states_delta_future,  # (n_future-1, N+M, state_dim)

            # ground truth information
            "state_future": obj_kp_future,  # (n_future, N, state_dim)

            # attr information
            "attrs": attrs,  # (N+M, attr_dim)
            "p_rigid": p_rigid,  # (n_instance,)
            "p_instance": p_instance,  # (N, n_instance)
            "obj_mask": obj_mask,  # (N,)

            "Rr": Rr,  # (max_nR, N)
            "Rs": Rs,  # (max_nR, N)

            "material_index": material_idx,  # (N, num_materials)
        }
        
        ## add physics parameters
        for material_name, material_dim in self.materials.items():
            if material_name in physics_param.keys():
                graph[material_name + '_physics_param'] = physics_param[material_name]
            else:
                graph[material_name + '_physics_param'] = torch.zeros(material_dim, dtype=torch.float32)

        return graph
    
if __name__ == "__main__":
    config_path = 'config/dynamics/rope.yaml'
    config = load_yaml(config_path)
    dataset_config = config['dataset_config']
    material_config = config['material_config']
    
    train_dataset = DynDataset(dataset_config, material_config, phase='train')
    graph = train_dataset[0]
    import pdb; pdb.set_trace()
    