import os
import glob
import numpy as np
import pickle

def load_pairs(pairs_path, episode_range, lowest_num=0):
    pair_lists = []
    for episode_idx in episode_range:
        # Increment episode index by the lowest starting episode number
        episode_idx = episode_idx + lowest_num
        n_pushes = len(list(glob.glob(os.path.join(pairs_path, f'{episode_idx:06}_*.txt'))))
        for push_idx in range(1, n_pushes+1):
            # Load the frame pairs if the file for that action's steps exists
            if not os.path.exists(os.path.join(pairs_path, f'{episode_idx:06}_{push_idx:02}.txt')):
                continue
            frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx:06}_{push_idx:02}.txt'))
            print(f"frame_pairs shape: {frame_pairs.shape}")
            if len(frame_pairs.shape) == 1: continue
            episodes = np.ones((frame_pairs.shape[0], 1)) * episode_idx
            pairs = np.concatenate([episodes, frame_pairs], axis=1) # (T, 8)
            pair_lists.extend(pairs)
    pair_lists = np.array(pair_lists).astype(int)
    return pair_lists

def load_dataset(dataset_config, material_config, phase='train'):
    # config
    data_name = dataset_config['data_name']
    if "data_folder" in dataset_config.keys():
        data_folder = dataset_config['data_folder']
    else:
        data_folder = data_name
    data_dir = os.path.join(dataset_config['data_dir'], data_folder)#+"_set_action_first_try_100_epochs")
    prep_dir = os.path.join(dataset_config['prep_data_dir'], data_folder)#+"_set_action_first_try_100_epochs")
    ratio = dataset_config['ratio']

    print("data dir for loading dataset: ", data_dir)
    print(f"prep_dir for loading dataset: {prep_dir}")
    
    # episodes
    epi_names = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()])
    lowest_num = int(epi_names[0])
    num_epis = len(epi_names)
    #num_epis = len(sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()]))
    print(f"Found number of episodes: {num_epis}. Lowest epi num: {lowest_num}")
    
    # load data pairs
    episode_range_phase = range(
        int(num_epis * ratio[phase][0]),
        int(num_epis * ratio[phase][1])
    )
    pairs_path = os.path.join(prep_dir, 'frame_pairs')
    pair_lists = load_pairs(pairs_path, episode_range_phase, lowest_num)
    print(f'{phase} dataset has {len(list(episode_range_phase))} episodes, {len(pair_lists)} frame pairs')
    
    # load physics params
    physics_params = []
    for episode_idx in range(num_epis):
        episode_idx = episode_idx + lowest_num
        physics_path = os.path.join(data_dir, f"{episode_idx:06}/property_params.pkl")
        with open(physics_path, 'rb') as f:
            properties = pickle.load(f)
        
        physics_params_episode = {}
        for material_name in dataset_config["materials"]:
            material_params = material_config[material_name]['physics_params']

            phys_norm_max = 1.0
            phys_norm_min = 0.0
            
            used_params = []
            for item in material_params:
                if item['name'] in properties.keys() and item['use']:
                    range_min = item['min']
                    range_max = item['max']
                    used_params.append((properties[item['name']] - range_min) / (range_max - range_min + 1e-6))
                    print(f"using param: {item['name']}, orig value: {properties[item['name']]}, normalized value: {used_params[-1]}")
            
            used_params = np.array(used_params).astype(np.float32)
            used_params = used_params * (phys_norm_max - phys_norm_min) + phys_norm_min
            physics_params_episode[material_name] = used_params
        
        physics_params.append(physics_params_episode)
        print("loaded physics params: ", physics_params)
    
    return pair_lists, physics_params, lowest_num

def get_position_paths(dataset_config):
    # Only save the paths to each episode and lazily load it in when needed
    data_name = dataset_config['data_name']
    if "data_folder" in dataset_config.keys():
        data_folder = dataset_config['data_folder']
    else:
        data_folder = data_name
    prep_dir = os.path.join(dataset_config['prep_data_dir'], data_folder)#+"_set_action_first_try_100_epochs")
    print(f"get_position_paths in load.py, prep_dir: {prep_dir}")
    positions_pkls = sorted(glob.glob(os.path.join(prep_dir, "*_positions.pkl")))
    return positions_pkls

def lazy_load_positions(positions_pkls, episode_idx):
    # Only save the paths to each episode and lazily load it in when needed
    # data_name = dataset_config['data_name']
    # if "data_folder" in dataset_config.keys():
    #     data_folder = dataset_config['data_folder']
    # else:
    #     data_folder = data_name
    # prep_dir = os.path.join(dataset_config['prep_data_dir'], data_folder)#+"_set_action_first_try_100_epochs")
    # print(f"lazy_load_positions in load.py, prep_dir: {prep_dir}")
    # positions_pkls = sorted(glob.glob(os.path.join(prep_dir, "*_positions.pkl")))
    # print("len of sorted positions pkl: ", len(positions_pkls))
    ## load positions
    # ['eef_pos', 'obj_pos']
    # eef_pos: (T, N_eef, 3)
    # obj_pos: (T, N_obj, 3)
    pkl_path = positions_pkls[episode_idx]
    # print(f"Epi index: {episode_idx}, loading in from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        positions = pickle.load(f)
    eef_pos = positions['eef_pos']
    obj_pos = positions['obj_pos']
    return eef_pos, obj_pos

def load_positions(dataset_config):
    ## config
    data_name = dataset_config['data_name']
    if "data_folder" in dataset_config.keys():
        data_folder = dataset_config['data_folder']
    else:
        data_folder = data_name
    prep_dir = os.path.join(dataset_config['prep_data_dir'], data_folder)#+"_set_action_first_try_100_epochs")
    print(f"load_positions in load.py, prep_dir: {prep_dir}")

    ## load positions
    # ['eef_pos', 'obj_pos', 'phys_params']
    # eef_pos: (n_epis, T, N_eef, 3)
    # obj_pos: (n_epis, T, N_obj, 3)
    # phys_params: (n_epis, 1)
    positions_path = os.path.join(prep_dir, 'positions.pkl')
    if os.path.exists(positions_path):
        with open(positions_path, 'rb') as f:
            positions = pickle.load(f) 
        eef_pos = positions['eef_pos'] 
        obj_pos = positions['obj_pos']
        return eef_pos, obj_pos
    else:
        # Load each episode's pickle
        eef_pos = []
        obj_pos = []
        positions_pkls = sorted(glob.glob(os.path.join(prep_dir, "*_positions.pkl")))
        print("sorted positions pkl: ", positions_pkls)
        # Sort in episodic order
        for idx,pkl_path in enumerate(positions_pkls):
            print(f"loading from {pkl_path}")
            with open(pkl_path, "rb") as f:
                positions = pickle.load(f)
            eef_step = positions['eef_pos']
            obj_step = positions['obj_pos']
            eef_pos.append(eef_step)
            obj_pos.append(obj_step)
        return eef_pos, obj_pos

def load_part_2_instance(dataset_config):
    ## config
    data_name = dataset_config['data_name']
    if "data_folder" in dataset_config.keys():
        data_folder = dataset_config['data_folder']
    else:
        data_folder = data_name
    prep_dir = os.path.join(dataset_config['prep_data_dir'], data_folder)
    # prep_dir = os.path.join(dataset_config['prep_data_dir'], data_folder)#+"_set_action_first_try_100_epochs")
    print(f"load part2obj in load.py, data_dir: {prep_dir}")
    ## load particle to object instance mapping
    # "part_2_obj_inst"
    p2o_path = os.path.join(prep_dir, 'part_2_obj_inst.pkl')
    if not os.path.exists(p2o_path):
        return None
    with open(p2o_path, "rb") as f:
        p2o = pickle.load(f)
    return p2o['part_2_obj_inst']

def load_part_inv_weight_is_0(dataset_config):
    data_name = dataset_config['data_name']
    if "data_folder" in dataset_config.keys():
        data_folder = dataset_config['data_folder']
    else:
        data_folder = data_name
    prep_dir = os.path.join(dataset_config['prep_data_dir'], data_folder)
    print(f"loading in particle inverse weight is 0 mask in load.py, data_dir: {prep_dir}")
    inv_weight_path = os.path.join(prep_dir, "particle_inv_weight_is_0.pkl")
    part_paths = sorted(glob.glob(os.path.join(prep_dir, "*_particle_inv_weight_is_0.pkl")))
    if not os.path.exists(inv_weight_path) and len(part_paths) == 0:
        return None
    if os.path.exists(inv_weight_path):
        with open(inv_weight_path, "rb") as f:
            inv = pickle.load(f)
        return inv['particle_inv_weight_is_0'] 
    else:
        inv = []
        for idx,part_path in enumerate(part_paths):
            print(f"loading from {part_path}")
            with open(part_path, "rb") as f:
                data = pickle.load(f)
            inv.append(data)
        return inv
    
