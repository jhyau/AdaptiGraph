import os
import glob
import numpy as np
import argparse
import pickle
import time

import sys
sys.path.append('.')
from sim.utils import load_yaml
from sim.data_gen.data import load_data
from dynamics.utils import quaternion_to_rotation_matrix

"""
Preprocess:
    - frame pairs
    - eef and object positions
    - physics params
    - metadata
"""

def read_in_filter_file(file_path):
    ## Returns a dict where the keys are episode numbers and values are list of actions that need to be filtered out
    result = {}
    with open(file_path, "r") as file:
        for line in file:
            if line.find("Final stats") != -1:
                # Done parsing through the flag file
                break
            
            if line.startswith("Episode: "):
                # f.write(f"Episode: {epi}, step/action: {step_idx}, max difference for same point at rest vs. at penultimate time step: {diff}\n")
                delim_comma = [x.strip() for x in line.split(",")]
                epi = delim_comma[0][delim_comma[0].find("Episode: ")+9:].strip()
                action = int(delim_comma[1][delim_comma[1].find("step/action: ")+13:].strip())
                print(f"epi: {epi}, action: {action}")
                if epi not in result:
                    result[epi] = [action]
                else:
                    result[epi].append(action)
    return result

    
def process_eef(eef_states, eef_dataset):
    """
    eef_states: (T, N_eef, 14)
    """
    T = eef_states.shape[0]
    if len(eef_states.shape) == 2:
        eef_states = eef_states.reshape(T, 1, 14)
    eef_pos = eef_dataset['pos']
    N_eef = len(eef_pos)

    out_eefs = np.zeros((T, eef_dataset['max_neef'], 3))   
    assert N_eef == eef_dataset['max_neef'], 'Number of eef not match.' 
    
    # process eef
    for i in range(T):
        for j in range(N_eef):
            if j >= eef_states.shape[1]:
                # granular case
                eef_idx = eef_states.shape[1] - 1
            else:
                eef_idx = j
            eef_state = eef_states[i][eef_idx]
            eef_pos_0 = eef_state[0:3]
            eef_quat = eef_state[6:10]
            eef_rot = quaternion_to_rotation_matrix(eef_quat)
            eef_final_pos = eef_pos_0 + np.dot(eef_rot, eef_pos[j])
            out_eefs[i, j] = eef_final_pos
    return out_eefs

def extract_physics(physics_path, obj):
    with open(physics_path, 'rb') as f:
        properties = pickle.load(f)
    # extract physics params
    if obj == 'rope':
        phys_param = np.array([
            properties['stiffness']
        ]).astype(np.float32)
    elif obj == 'granular':
        phys_param = np.array([
            properties['granular_scale']
        ])
    elif obj == 'cloth':
        phys_param = np.array([
            properties['sf']
        ])
    elif obj == 'softbody':
        phys_param = np.array([
            properties['stiffness']
        ])
    elif obj == 'bunnybath':
        phys_param = np.array([
            properties['viscosity']
        ])
    elif obj == 'multiobj':
        phys_param = np.array([
            properties['stiffness']
            #properties['stiffness2']
        ])
    else:
        raise ValueError('Invalid object type.')
    return phys_param

def extract_push(eef, dist_thresh, n_his, n_future, n_frames, store_rest_state):
    """
    eef: (T, N_eef, 3)
    """
    T = eef.shape[0]
    eef = eef[:, 0] # (T, 3)
    
    # generate start-end pair
    frame_idxs = []
    cnt = 0
    start_frame = 0
    end_frame = T
    for fj in range(T):
        curr_frame = fj
        
        # search backward (n_his)
        frame_traj = [curr_frame]
        eef_curr = eef[curr_frame]
        fi = fj
        while fi >= start_frame:
            eef_fi = eef[fi]
            x_curr, y_curr, z_curr = eef_curr[0], eef_curr[1], eef_curr[2] #x, z only before. also take y into account
            x_fi, y_fi, z_fi = eef_fi[0], eef_fi[1], eef_fi[2]
            dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2 + (y_curr - y_fi) ** 2)
            if dist_curr >= dist_thresh:
                frame_traj.append(fi)
                eef_curr = eef_fi
            fi -= 1
            if store_rest_state and len(frame_traj) == n_his - 1:
                # if storing rest state, then prepend each frame trajectory with 0 for the first frame in history
                frame_traj.append(0)
            if len(frame_traj) == n_his:
                break
        else:
            # pad to n_his
            frame_traj = frame_traj + [frame_traj[-1]] * (n_his - len(frame_traj))
        frame_traj = frame_traj[::-1] # Reverses the list
        
        # search forward (n_future)
        eef_curr = eef[curr_frame]
        fi = fj
        while fi < end_frame:
            eef_fi = eef[fi]
            x_curr, y_curr, z_curr = eef_curr[0], eef_curr[1], eef_curr[2]
            x_fi, y_fi, z_fi = eef_fi[0], eef_fi[1], eef_fi[2]
            dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (z_curr - z_fi) ** 2 + (y_curr - y_fi) ** 2)
            if dist_curr >= dist_thresh:
                frame_traj.append(fi)
                eef_curr = eef_fi
            fi += 1
            if len(frame_traj) == n_his + n_future:
                cnt += 1
                break
        else:
            # pad to n_future
            frame_traj = frame_traj + [frame_traj[-1]] * (n_his + n_future - len(frame_traj))
            cnt += 1
        
        frame_idxs.append(frame_traj)
        
        # push centered
        if fj == end_frame - 1:
            frame_idxs = np.array(frame_idxs)
            if store_rest_state:
                # The first index should be rest state, frame index 0
                print(f"shape of frame_idxs: {frame_idxs.shape} and n_frames: {n_frames}")
                frame_idxs[:, 1:] = frame_idxs[:, 1:] + n_frames
            else:
                frame_idxs = frame_idxs + n_frames # add previous steps
    
    return frame_idxs, cnt

def preprocess(config, lazy_loading):
    time_start = time.time()
    
    # config
    dataset_config = config['dataset_config']
    data_name = dataset_config['data_name']
    eef_dataset = dataset_config['eef']
    
    if "data_folder" in dataset_config.keys():
        data_folder = dataset_config['data_folder']
    else:
        data_folder = data_name

    data_dir = os.path.join(dataset_config['data_dir'], data_folder)
    save_dir = os.path.join(dataset_config['prep_data_dir'], data_folder)
    push_save_dir = os.path.join(save_dir, 'frame_pairs')
    os.makedirs(push_save_dir, exist_ok=True)
    print(f"preprocess save dir: {save_dir}")
    
    n_his = dataset_config['n_his']
    n_future = dataset_config['n_future']
    dist_thresh = dataset_config['dist_thresh']
    store_rest_state = dataset_config['store_rest_state']

    # if store_rest_state:
    #     # Need to subtract one from n_his for preprocessing since one step is storing the rest state
    #     n_his = n_his - 1
    
    # File of actions to be filtered out
    filter_file = os.path.join(data_dir, "filter_unwanted_flex_artifacts.txt")
    filter_file_exists = os.path.exists(filter_file)
    if filter_file_exists:
        filter_dict = read_in_filter_file(filter_file)

    # episodes
    epi_list = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()])
    num_epis = len(epi_list)
    print(f"Preprocessing starts. Number of episodes: {num_epis}.")
    
    # preprocessing
    all_eef_pos = [] # (n_epis, N_eef, 3) 
    all_obj_pos = [] # (n_epis, N_obj, 3)
    phys_params = [] # (n_epis, N_phy, 1)
    #all_part_2_obj_instance = [] # (n_epis, N_obj, 1)
    all_part_inv_weight_is_0 = [] # (n_epis, N_obj, 1)
    for epi_idx, epi in enumerate(epi_list):
        epi_time_start = time.time()
        
        epi_dir = os.path.join(data_dir, epi)
        
        # preprocess property params
        physics_path = os.path.join(epi_dir, 'property_params.pkl')
        phys_param = extract_physics(physics_path, data_name)
        phys_params.append(phys_param)
        
        # preprocess step info
        num_steps = len(list(glob.glob(os.path.join(epi_dir, '*.h5')))) - 1
        
        eef_steps, obj_steps, part_2_obj_steps, part_inv_weight_0_steps = [], [], [], []
        n_frames = 0
        for step_idx in range(1, num_steps+1):
            # extract data
            data_path = os.path.join(epi_dir, f'{step_idx:02}.h5')
            data = load_data(data_path) # ['action', 'eef_states', 'info', 'observations', 'positions', 'part_2_obj_inst']
            
            eef_states = data['eef_states'] # (T, N_eef, 14)
            positions = data['positions'] # (T, N_obj, 3)
            #if "part_2_obj_inst" in data.keys():
            #    part_2_obj_inst = data['part_2_obj_inst'] # (T, N_obj, 1)
            #    
            #    # particle to object instance mapping
            #    part_2_obj_steps.append(part_2_obj_inst)
            
            if "particle_inv_weight_is_0" in data.keys():
                # boolean mask that's true when a particle's inverse weight is 0 (fixed in place)
                part_in_weight_0 = data['particle_inv_weight_is_0']
                part_inv_weight_0_steps.append(part_in_weight_0)

            # preprocess eef and push
            out_eef = process_eef(eef_states, eef_dataset) # (T, N_eef, 3)
            frame_idxs, cnt = extract_push(out_eef, dist_thresh, n_his, n_future, n_frames, store_rest_state)
            assert len(frame_idxs) == cnt, 'Number of pushes not match.'
            n_frames += cnt
            
            # eef and object positions
            eef_steps.append(out_eef)
            obj_steps.append(positions)

            ## If the current action of the episode is flagged, then skip this action and do not write to savetxt
            if filter_file_exists:
                if epi in filter_dict and step_idx in filter_dict[epi]:
                    continue
            
            # save frame idxs
            np.savetxt(os.path.join(push_save_dir, f'{epi}_{(step_idx):02}.txt'), frame_idxs, fmt='%d')
            print(f"Preprocessed episode {epi_idx+1}/{num_epis}, step {step_idx}/{num_steps}: Number of pushes {cnt}.")
        
        eef_steps = np.concatenate(eef_steps, axis=0)
        obj_steps = np.concatenate(obj_steps, axis=0)
        assert eef_steps.shape[0] == obj_steps.shape[0] == n_frames
        if not lazy_loading:
            all_eef_pos.append(eef_steps)
            all_obj_pos.append(obj_steps)
        
        #if len(part_2_obj_steps) > 0:
        #    part_2_obj_steps = np.concatenate(part_2_obj_steps, axis=0)
        #    all_part_2_obj_instance.append(part_2_obj_steps)
        #    assert eef_steps.shape[0] == obj_steps.shape[0] == n_frames == part_2_obj_steps.shape[0]
        
        if len(part_inv_weight_0_steps) > 0:
            part_inv_weight_0_steps = np.concatenate(part_inv_weight_0_steps, axis=0)
            assert eef_steps.shape[0] == obj_steps.shape[0] == n_frames == part_inv_weight_0_steps.shape[0]
            if not lazy_loading:
                all_part_inv_weight_is_0.append(part_inv_weight_0_steps)
            else:
                print(f"Lazy loading, so saving everything in separate episodes...")
                inv_weight_path = os.path.join(save_dir, f"{epi}_particle_inv_weight_is_0.pkl")
                inv_weight_info = {
                    'particle_inv_weight_is_0': part_inv_weight_0_steps
                }
                with open(inv_weight_path, "wb") as f:
                    pickle.dump(inv_weight_info, f)

        if lazy_loading:
            # Write out positions for each individual episode (OOM issue when trying to save all at once for 1000 epis)
            # save eef and object positions
            # (num_steps, n_particles, 3)
            pos_path = os.path.join(save_dir, f'{epi}_positions.pkl')
            pos_info = {
                'eef_pos': eef_steps, 
                'obj_pos': obj_steps,
            }
            with open(pos_path, 'wb') as f:
                pickle.dump(pos_info, f)
            # assert len(all_eef_pos) == len(all_obj_pos) == num_epis

        epi_time_end = time.time()
        print(f'Episode {epi_idx+1}/{num_epis} has frames {obj_steps.shape[0]} took {epi_time_end - epi_time_start:.2f}s.')
    
    # save physics params
    phys_params = np.stack(phys_params, axis=0)
    phys_params_max = np.max(phys_params, axis=0)
    phys_params_min = np.min(phys_params, axis=0)
    phys_params_range = np.stack([phys_params_min, phys_params_max], axis=0)
    print(f"Physics params range: {phys_params_range}")
    np.savetxt(os.path.join(save_dir, 'phys_range.txt'), phys_params_range)
    
    if not lazy_loading:
        # save eef and object positions
        pos_path = os.path.join(save_dir, 'positions.pkl')
        pos_info = {
        'eef_pos': all_eef_pos, 
        'obj_pos': all_obj_pos,
        }
        with open(pos_path, 'wb') as f:
            pickle.dump(pos_info, f)
        assert len(all_eef_pos) == len(all_obj_pos) == num_epis

    # save particle to object instance mapping
    #if len(all_part_2_obj_instance) > 0:
    #    p2o_path = os.path.join(save_dir, "part_2_obj_inst.pkl")
    #    p2o_info = {
    #        'part_2_obj_inst': all_part_2_obj_instance
    #    }
    #    with open(p2o_path, "wb") as f:
    #        pickle.dump(p2o_info, f)
    #    assert len(all_part_2_obj_instance) == num_epis
    
    if not lazy_loading:
        # save boolean mask of particles with inverse weight 0
        if len(all_part_inv_weight_is_0) > 0:
            inv_weight_path = os.path.join(save_dir, "particle_inv_weight_is_0.pkl")
            inv_weight_info = {
                'particle_inv_weight_is_0': all_part_inv_weight_is_0
            }
            with open(inv_weight_path, "wb") as f:
                pickle.dump(inv_weight_info, f)
            assert len(all_part_inv_weight_is_0) == num_epis

    # save metadata
    with open(os.path.join(save_dir, 'metadata.txt'), 'w') as f:
        f.write(f'{dist_thresh},{n_future},{n_his}')
    
    time_end = time.time()
    print(f"Preprocessing finished for Episodes {num_epis}. Time taken: {time_end - time_start:.2f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/dynamics/rope.yaml')
    parser.add_argument("--lazy_loading", action="store_true", help="Set this flag to true to save each episode's position and inverse particle weight is 0 mask separately")
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    
    preprocess(config, args.lazy_loading)
