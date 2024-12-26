import os
import glob
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import cv2

import sys
sys.path.append('.')
from dynamics.gnn.model import DynamicsPredictor
from sim.utils import load_yaml
from dynamics.utils import set_seed, truncate_graph, pad, pad_torch
from dynamics.dataset.load import load_dataset, load_positions, load_part_2_instance, load_part_inv_weight_is_0
from dynamics.rollout.graph import construct_graph, get_next_pair_or_break_episode, get_next_pair_or_break_episode_pushes
from dynamics.rollout.graph import extract_imgs, visualize_graph, moviepy_merge_video
from dynamics.dataset.graph import construct_edges_from_states

def rollout_from_start_graph(graph, fps_idx_list, dataset_config, material_config,
                             model, device, eef_pos, obj_pos,
                             current_start, current_end, get_next_pair_or_break_func,
                             pairs, save_dir, viz, imgs, cam_info, part_2_obj_inst, 
                             part_inv_weight_0, hetero):

    obj_mask = graph['obj_mask'].numpy()
    obj_kp_num = obj_mask.sum()
    eef_kp_num = eef_pos.shape[1]
    
    dataset = dataset_config['datasets'][0]
    max_nobj = dataset['max_nobj']
    max_nR = dataset['max_nR']
    adj_thresh = (dataset['adj_radius_range'][0] + dataset['adj_radius_range'][1]) / 2
    topk = dataset['topk']
    connect_tool_all = dataset['connect_tool_all']
    if "connect_tool_surface" in dataset:
        connect_tool_surface = dataset['connect_tool_surface']
        connect_tool_surface_ratio = dataset['connect_tool_surface_ratio']
        print(f"connecting tool to surface: {connect_tool_surface}, surface ratio: {connect_tool_surface_ratio}")
    else:
        connect_tool_surface = False
        connect_tool_surface_ratio = 1.0
    
    n_his = dataset_config['n_his']
    n_frames = obj_pos.shape[0]
    assert eef_pos.shape[0] == n_frames
    
    ## viz initial graph
    if viz:
        # imgs: (T, H, W, 3)
        # Rr, Rs: (n_R, N)
        # eef_kp: (2, N_eef, 3)
        # kp_vis: (N_fps, 3)
        # cam_info: {'cam', 'cam_extr', 'cam_intr'}
        Rr = graph['Rr'].numpy()
        Rs = graph['Rs'].numpy()
        eef_kp = graph['eef_kp'].numpy()
        kp_vis = graph['state'][-1, :obj_kp_num].numpy()
        print(f"obj_kp_num: {obj_kp_num}, obj_mask size: {obj_mask.shape}, kp_vis shape: {kp_vis.shape}")
        
        save_dir = os.path.join(save_dir, f"cam_{cam_info['cam']}")
        print(f"save_dir to be created: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

        physics_param = None
        for name in graph.keys():
            if name.endswith('_physics_param'):
                physics_param = graph[name][:obj_kp_num].numpy()
        
        if part_2_obj_inst is not None:
            vis_part_2_obj = part_2_obj_inst[0][fps_idx_list]
        else:
            vis_part_2_obj = None
        
        if part_inv_weight_0 is not None:
            vis_part_inv_weight_0 = part_inv_weight_0[0][fps_idx_list]
        else:
            vis_part_inv_weight_0 = None
        pred_kp_proj_last, gt_kp_proj_last, gt_lineset, pred_lineset = \
            visualize_graph(imgs, cam_info, kp_vis, kp_vis, eef_kp, Rr, Rs,
                            current_start, current_end, 0, save_dir, max_nobj, 
                            part_2_obj_inst=vis_part_2_obj, 
                            part_inv_weight_0=vis_part_inv_weight_0,
                            physics_param=physics_param, hetero=hetero)

    ## prepare graph
    graph = {key: graph[key].unsqueeze(0).to(device) for key in graph.keys()}
    
    ## iterative rollout
    rollout_steps = 100
    error_list = []
    idx_list = [[current_start, current_end]]
    
    with torch.no_grad(): # eval
        for i in range(1, 1 + rollout_steps):
            print(f"rollout index: {i}")
            # prediction
            graph = truncate_graph(graph) # remove the padding
            pred_state, pred_motion = model(**graph)
            pred_state = pred_state.detach().cpu().numpy() # (1, N_obj, 3)
            
            # prepare gt
            gt_state_all = obj_pos[current_end] # (N_obj_all, 3)
            fps_gt_state = gt_state_all[fps_idx_list] # (N_fps, 3)
            gt_state = pad(fps_gt_state, max_nobj) # (N_obj, 3)
            
            # next step input
            obj_kp = pred_state[0][obj_mask] 
            gt_kp = gt_state[obj_mask] 
            
            # fps for visualization
            obj_kp_vis = obj_kp[:obj_kp_num] # (N_fps, 3)
            gt_kp_vis = gt_kp[:obj_kp_num] # (N_fps, 3)
            if part_2_obj_inst is not None:
                part_2_obj_inst_vis = part_2_obj_inst[current_end][fps_idx_list]
            else:
                part_2_obj_inst_vis = None
            print(f"obj_kp_num: {obj_kp_num}, obj_kp shape: {obj_kp.shape}, obj_kp_vis shape: {obj_kp_vis.shape}")
            max_y = np.max(obj_kp_vis[:,1]) * connect_tool_surface_ratio #0.8
            
            if part_inv_weight_0 is not None:
                part_inv_weight_0_vis = part_inv_weight_0[current_end][fps_idx_list]
            else:
                part_inv_weight_0_vis = None

            # calculate error
            error = np.linalg.norm(obj_kp - gt_kp, axis=-1).mean()
            error_list.append(error)
            print(f"error from rollout step {i}: {error}")
            
            # prepare next pair, start, end
            next_pair = get_next_pair_or_break_func(pairs, n_his, n_frames, current_end)
            if next_pair is None: break
            current_start = next_pair[n_his-1]
            current_end = next_pair[n_his]
            idx_list.append([current_start, current_end])
            
            # generate next graph based on the current prediction
            eef_kp_start = eef_pos[current_start] # (N_eef, 3)
            eef_kp_end = eef_pos[current_end] # (N_eef, 3)
            eef_kp = np.concatenate([eef_kp_start, eef_kp_end], axis=0) 
            eef_kp = eef_kp.reshape(2, eef_pos.shape[1], eef_pos.shape[2]) # (2, N_eef, 3)
            
            states = np.concatenate([pred_state[0], eef_kp[0]], axis=0) # (N_obj + N_eef, 3)
            
            states_delta = np.zeros_like(states, dtype=np.float32) # (N_obj + N_eef, 3)
            states_delta[max_nobj : max_nobj + eef_kp_num] = eef_kp_end - eef_kp_start
            
            Rr, Rs = construct_edges_from_states(torch.tensor(states), adj_thresh,
                                                 mask=graph['state_mask'][0],
                                                 tool_mask=graph['eef_mask'][0],
                                                 topk=topk, connect_tools_all=connect_tool_all,
                                                 max_y=max_y,
                                                 connect_tools_surface=connect_tool_surface)
            print(f"max_nR: {max_nR}, Rr size: {Rr.size()}, Rs size: {Rs.size()}")
            Rr = pad_torch(Rr, max_nR)
            Rs = pad_torch(Rs, max_nR)
            
            state_history = graph['state'][0].detach().cpu().numpy()
            state_history = np.concatenate([state_history[1:], states[None]], axis=0)
            
            new_graph = {
                "state": torch.from_numpy(state_history).unsqueeze(0).to(device).float(),  # (n_his, N+M, state_dim)
                "action": torch.from_numpy(states_delta).unsqueeze(0).to(device).float(),  # (N+M, state_dim)
                
                "Rr": Rr.unsqueeze(0).to(device).float(),  # (n_rel, N+M)
                "Rs": Rs.unsqueeze(0).to(device).float(),  # (n_rel, N+M)
                
                "attrs": graph["attrs"],  # (N+M, attr_dim)
                "p_rigid": graph["p_rigid"],  # (n_instance,)
                "p_instance": graph["p_instance"],  # (N, n_instance)
                "obj_mask": graph["obj_mask"],  # (N,)
                "eef_mask": graph["eef_mask"],  # (N+M,)
                "state_mask": graph["state_mask"],  # (N+M,)
                "material_index": graph["material_index"],  # (N, num_materials)
            }
            mat_name = None
            for name in graph.keys():
                if name.endswith('_physics_param'):
                    mat_name = name
                    new_graph[name] = graph[name]
            
            graph = new_graph
            
            ## viz graph
            if viz:
                pred_kp_proj_last, gt_kp_proj_last, gt_lineset, pred_lineset = \
                visualize_graph(imgs, cam_info, obj_kp_vis, gt_kp_vis, eef_kp, Rr, Rs,
                                current_start, current_end, i, save_dir, max_nobj,
                                gt_lineset=gt_lineset, pred_lineset=pred_lineset,
                                pred_kp_proj_last=pred_kp_proj_last, gt_kp_proj_last=gt_kp_proj_last,
                                part_2_obj_inst=part_2_obj_inst_vis,
                                part_inv_weight_0=part_inv_weight_0_vis,
                                physics_param=graph[mat_name][:, :obj_kp_num].detach().cpu().numpy(),
                                hetero=hetero)
                
    return error_list

def rollout_episode_pushes(model, device, dataset_config, material_config,
                        eef_pos, obj_pos, episode_idx, pairs, physics_param,
                        save_dir, viz, imgs, cam_info, part_2_obj_inst, part_inv_weight_0,
                        keep_prev_fps, hetero):
    n_his = dataset_config['n_his']
    
    ## get steps
    #pairs_path = os.path.join(dataset_config['prep_data_dir'], dataset_config['data_name']+"_set_action_first_try_100_epochs", 'frame_pairs')
    print(f"episode index: {episode_idx}")
    data_name = dataset_config['data_name']
    if "data_folder" in dataset_config.keys():
        data_folder = dataset_config['data_folder']
    else:
        data_folder = data_name
    pairs_path = os.path.join(dataset_config['prep_data_dir'], data_folder, 'frame_pairs')
    pairs_list = sorted(list(glob.glob(os.path.join(pairs_path, f'{episode_idx:06}_*.txt'))))
    num_steps = len(pairs_list)
    print(f"num steps: {num_steps}, pairs_path: {pairs_path}")
    
    error_list_pushes = []
    prev_fps_idx_list = None
    fps2phys = None
    print(f"pairs list: {pairs_list}")
    for i in range(num_steps):
        valid_pairs = np.loadtxt(pairs_list[i]).astype(int)
        pair = valid_pairs[0] 
        start = pair[n_his-1]
        end = pair[n_his]
        print(f"pair: {pair}, start: {start}, end: {end}")
        
        eef_pos_epi = eef_pos[episode_idx] # (T, N_eef, 3)
        obj_pos_epi = obj_pos[episode_idx] # (T, N_obj, 3)
        # Get particle to object instance mapping
        if part_2_obj_inst:
            part_2_obj_inst_epi = part_2_obj_inst[episode_idx] # (T, N_obj, 1)
            print(f"obj_pos_epi shape: {obj_pos_epi.shape}, part2obj shape: {part_2_obj_inst_epi.shape}")
        else:
            part_2_obj_inst_epi = None
    
        # get bool mask of particles that have inverse weight of 0
        if part_inv_weight_0:
            part_inv_weight_0_epi = part_inv_weight_0[episode_idx] # (T, N_obj, 1)
            print(f"particle inv weight is 0 shape: {part_inv_weight_0_epi.shape}")
        else:
            part_inv_weight_0_epi = None

        ## construct graph
        graph, fps_idx_list, fps2phys = construct_graph(dataset_config, material_config, eef_pos_epi, obj_pos_epi,
                                        n_his, pair, physics_param, part_2_obj_inst=part_2_obj_inst_epi,
                                        prev_fps_idx_list=prev_fps_idx_list, fps2phys=fps2phys,
                                        hetero=hetero)
    
        # Use the same fps indices after the first sampling
        if keep_prev_fps:
            print("using the same fps index list")
            prev_fps_idx_list = fps_idx_list
            fps2phys = fps2phys
        ## rollout from start
        error_list = rollout_from_start_graph(graph, fps_idx_list, dataset_config, material_config,
                                          model, device, eef_pos_epi, obj_pos_epi,
                                          start, end, get_next_pair_or_break_episode_pushes,
                                          pairs, save_dir, viz, imgs, cam_info, part_2_obj_inst_epi, 
                                          part_inv_weight_0_epi, hetero)
        
        error_list_pushes.append(error_list)
        
        ## plot error
        plt.figure(figsize=(10, 5))
        plt.plot(error_list)
        plt.xlabel('time step')
        plt.ylabel('error')
        plt.grid()
        plt.savefig(os.path.join(save_dir, f'error_{i+1}.png'), dpi=300)
        plt.close()
    
        error_list = np.array(error_list)
        np.savetxt(os.path.join(save_dir, f'error_{i+1}.txt'), error_list)
        
    ## visualization
    if viz:
        img_path = os.path.join(save_dir, f"cam_{cam_info['cam']}")
        fps = 10
        pred_out_path = os.path.join(img_path, "pred.mp4")
        moviepy_merge_video(img_path, 'pred', pred_out_path, fps)
        gt_out_path = os.path.join(img_path, "gt.mp4")
        moviepy_merge_video(img_path, 'gt', gt_out_path, fps)
        both_out_path = os.path.join(img_path, "both.mp4")
        moviepy_merge_video(img_path, 'both', both_out_path, fps)
    
    return error_list_pushes

def rollout_dataset(model, device, config, save_dir, viz, keep_prev_fps, hetero):
    ## config
    dataset_config = config['dataset_config']
    material_config = config['material_config']
    
    ## load pair lists
    pair_lists, physics_params = load_dataset(dataset_config, material_config, phase='valid')
    pair_lists = np.array(pair_lists)
    print(f"Rollout dataset has {len(pair_lists)} frame pairs.")
    
    ## load positions
    eef_pos, obj_pos = load_positions(dataset_config)

    ## load particle to object instance mapping
    part_2_obj_inst = load_part_2_instance(dataset_config)

    ## load bool mask of particles with inverse weight 0
    part_inv_weight_0 = load_part_inv_weight_is_0(dataset_config)
    
    ## rollout
    total_error_short = []
    
    ## get errors for each episode
    episode_idx_list = sorted(list(np.unique(pair_lists[:, 0]).astype(int))) # [7]
    for episode_idx in episode_idx_list:
        pair_lists_episode = pair_lists[pair_lists[:, 0] == episode_idx][:, 1:]
        physics_params_episode = physics_params[episode_idx]
        
        if viz:
            imgs, cam_info = extract_imgs(dataset_config, episode_idx, cam=0)
            print(f"num imgs: {len(imgs)}, num pair lists episode: {len(pair_lists_episode)}")
            assert len(imgs) == len(pair_lists_episode)
            # Save the original images and make into movie
            print(f"saving OG images...")
            og_img_path = os.path.join(save_dir, f"og_epi_{episode_idx}")
            os.makedirs(og_img_path, exist_ok=True)
            pred_out_path = os.path.join(og_img_path, f"epi_{episode_idx}_og.mp4")
            fps = 10
            for i,img in enumerate(imgs):
                cv2.imwrite(os.path.join(og_img_path, f'epi_{episode_idx}_step_{i}_og.jpg'), img)
            moviepy_merge_video(og_img_path, 'og', pred_out_path, fps)
        else:
            imgs, cam_info = None, None
        
        save_dir_episode_pushes = os.path.join(save_dir, f"{episode_idx}", "short")
        os.makedirs(save_dir_episode_pushes, exist_ok=True)
        error_list_short = rollout_episode_pushes(model, device, dataset_config, material_config,
                                        eef_pos, obj_pos, episode_idx,
                                        pair_lists_episode, physics_params_episode,
                                        save_dir_episode_pushes, viz, imgs, cam_info, part_2_obj_inst, 
                                        part_inv_weight_0,
                                        keep_prev_fps, hetero)
        total_error_short.extend(error_list_short)
    
    ## final statistics
    for (total_error, save_name) in zip([total_error_short], ['error_short']):
        max_step = max([len(total_error[i]) for i in range(len(total_error))])
        min_step = min([len(total_error[i]) for i in range(len(total_error))])
        step_error = np.zeros((min_step, len(total_error)))
        for i in range(min_step):
            for j in range(len(total_error)):
                step_error[i, j] = total_error[j][i]

        # vis error
        # step_mean_error = step_error.mean(1)
        np.savetxt(os.path.join(save_dir, f'{save_name}.txt'), step_error)

        # Calculate the median, 75th percentile, and 25th percentile
        median_error = np.median(step_error, axis=1)
        step_75_error = np.percentile(step_error, 75, axis=1)
        step_25_error = np.percentile(step_error, 25, axis=1)

        # plot error
        plt.figure(figsize=(10, 5))
        plt.plot(median_error)
        plt.xlabel("time step")
        plt.ylabel("error")
        plt.grid()

        ax = plt.gca()
        x = np.arange(median_error.shape[0])
        ax.fill_between(x, step_25_error, step_75_error, alpha=0.2)

        plt.savefig(os.path.join(save_dir, f'{save_name}.png'), dpi=300)
        plt.close()

def rollout(config, epoch, ckpt_data_name, viz=False, keep_prev_fps=False, hetero=False):
    ## config
    dataset_config = config['dataset_config']
    train_config = config['train_config']
    model_config = config['model_config']
    material_config = config['material_config']
    rollout_config = config['rollout_config']
    
    set_seed(train_config['random_seed'])
    device = torch.device(dataset_config['device'])
    
    data_name = dataset_config['data_name']
    if ckpt_data_name is None:
        ckpt_data_name = data_name
    out_dir = os.path.join(rollout_config['out_dir'])
    if "output_name" in dataset_config:
        save_dir = os.path.join(out_dir, f'rollout-{dataset_config["output_name"]}-model_{epoch}')
        #print("output_name save dir: ", save_dir)
    else:
        save_dir = os.path.join(out_dir, f'rollout-{data_name}-model_{epoch}')
    print(f"save dir: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    if epoch == 'latest':
        #checkpoint_dir = os.path.join(train_config['out_dir'], data_name, 'checkpoints', 'latest.pth')
        checkpoint_dir = os.path.join(train_config['out_dir'], ckpt_data_name, 'checkpoints', 'latest.pth')
    else:
        #checkpoint_dir = os.path.join(train_config['out_dir'], data_name, 'checkpoints', 'model_100.pth')
        #checkpoint_dir = os.path.join(train_config['out_dir'], data_name, 'checkpoints', 'model_{}.pth'.format(epoch))
        checkpoint_dir = os.path.join(train_config['out_dir'], ckpt_data_name, 'checkpoints', 'model_{}.pth'.format(epoch))
    
    print("checkpoint_dir: ", checkpoint_dir) 
    ## load model
    model = DynamicsPredictor(model_config, 
                              material_config,
                              dataset_config,
                              device)
    model.to(device)
    
    mse_loss = torch.nn.MSELoss()
    loss_funcs = [(mse_loss, 1)]
    
    model.eval()
    model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
    
    ## rollout dataset
    rollout_dataset(model, device, config, save_dir, viz, keep_prev_fps, hetero)
    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/dynamics/rope.yaml')
    arg_parser.add_argument('--ckpt_data_name', type=str, default=None)
    arg_parser.add_argument('--epoch', type=str, default='100')
    arg_parser.add_argument('--viz', action='store_true')
    arg_parser.add_argument('--keep_prev_fps', action='store_true') # set this flag to keep FPS indices for all time steps
    arg_parser.add_argument('--hetero', action='store_true') # set this flag to change to hetero phys params for rollout
    args = arg_parser.parse_args()

    config = load_yaml(args.config)
    
    rollout(config, args.epoch, args.ckpt_data_name, args.viz, keep_prev_fps=args.keep_prev_fps, hetero=args.hetero)
