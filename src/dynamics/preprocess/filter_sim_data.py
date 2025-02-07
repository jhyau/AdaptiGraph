import argparse
import os, sys
import time
import h5py
import numpy as np

from sim.utils import load_yaml
from sim.data_gen.data import load_data
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/dynamics/rope.yaml')
parser.add_argument('--output_file_name', type=str, default=None)
args = parser.parse_args()

dataset_config = load_yaml(args.config)

## Path to the simulation data to filter
data_folder = dataset_config['data_folder']
data_dir = os.path.join(dataset_config['data_dir'], data_folder)

if parser.output_file_name is None:
    out_file = os.path.join(data_dir, "filter_unwanted_flex_artifacts.txt")
else:
    out_file = os.path.join(data_dir, dataset_config['filter_file_name'])
# os.makedirs(push_save_dir, exist_ok=True)
print(f"filtering output file path: {out_file}")

# Keep track of which episode and which actions need to be filtered out/flagged
f = open(out_file, "w")

# episodes
epi_list = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.isdigit()])
num_epis = len(epi_list)

# Iterate through all episodes
for epi_idx, epi in tqdm(enumerate(epi_list)):
    epi_time_start = time.time()
    
    epi_dir = os.path.join(data_dir, epi)
    num_steps = len(list(glob.glob(os.path.join(epi_dir, '*.h5')))) - 1

    # Get the 0th action object particle positions (1, N_obj, 3)
    rest_state = load_data(os.path.join(epi_dir, f"00.h5"))
    rest_state_positions = rest_state["positions"]
    num_particles = rest_state.shape[1]

    # Threshold (for stiff cases at least)
    THRES = 0.1 #10.0

    # Iterate through all actions for this episode
    for step_idx in range(1, num_steps+1):
        # extract data
        data_path = os.path.join(epi_dir, f'{step_idx:02}.h5')
        data = load_data(data_path) # ['action', 'eef_states', 'info', 'observations', 'positions', 'part_2_obj_inst']

        eef_states = data['eef_states'] # (T, N_eef, 14)
        positions = data['positions'] # (T, N_obj, 3)

        # Calculate difference between the positions of first time step of the action with the 0th
        #dist_first_time_step_to_rest = np.sum(positions[0,:,:] - rest_state_positions[0,:,:])

        # Calculate difference between the positions of second-to-last time step of the action with the 0th
        #dist_last_time_step_to_rest = np.sum(positions[-1,:,:] - rest_state_positions[0,:,:])

        # Find the MAX distance between a point at rest vs. at second to last time step
        # Also check distance of every object point to the end effector points, if distance is small then remove
        diff = np.max(np.abs(data['positions'][-2,:,:] - rest_state_positions[0,:,:])) #dist_first_time_step_to_rest - dist_last_time_step_to_rest
        if np.max(np.abs(data['positions'][-2,:,:] - rest_state_positions[0,:,:])) > THRES:
            f.write(f"Episode: {epi}, step/action: {step_idx}, max difference for same point at rest vs. at penultimate time step: {diff}")

# Close file
f.close()
print(f"Filtering complete")