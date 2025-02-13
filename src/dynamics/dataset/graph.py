import numpy as np
import copy
import torch

from dgl.geometry import farthest_point_sampler
from dynamics.utils import fps_rad_idx

def fps(obj_kp_start, max_nobj, fps_radius_range, verbose=False):
        ## farthest point sampling
        particle_tensor = torch.from_numpy(obj_kp_start).float().unsqueeze(0) # [1, N, 3]
        fps_idx_tensor = farthest_point_sampler(particle_tensor, min(max_nobj, particle_tensor.shape[1]),
                                                start_idx=np.random.randint(0, particle_tensor.shape[1]))[0]
        fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32) # (max_nobj, )
        
        ## downsample to uniform radius
        downsample_particle = particle_tensor[0, fps_idx_1].numpy() # (max_nobj, 3)
        
        ## choose fps radius
        if type(fps_radius_range) == float:
            fps_radius = fps_radius_range
        elif len(fps_radius_range) == 2:
            fps_radius = np.random.uniform(fps_radius_range[0], fps_radius_range[1])
        else:
            raise ValueError(f"Invalid fps_radius_range: {fps_radius_range}.")
        
        _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
        fps_idx_2 = fps_idx_2.astype(np.int32)
        fps_idx = fps_idx_1[fps_idx_2]
        
        if verbose:
            print(f"FPS num particles: {len(fps_idx)} with index list \n {fps_idx}. \n")
        
        # fps index list
        fps_idx_list = np.array(fps_idx)  # (n_fps, )
        
        return fps_idx_list

def determine_closest_plane(max_y, min_x, max_x, min_z, max_z):
    ## Given distance values to each respective plane, returns the shortest distance
    values = [max_y, min_x, max_x, min_z, max_z]
    name = ["max_y", "min_x", "max_x", "min_z", "max_z"]
    indices = np.argsort(values)
    print(f"the squared sum distances to each plane: {values}")
    print(f"closest plane: {name[indices[0]]} and value: {values[indices[0]]}")
    return name[indices[0]], values[indices[0]], name[indices[1]], values[indices[1]]

def conditions_for_tool_to_surface(plane_name, max_y, max_x, max_z, min_x, min_z, s_receiv, s_sender):
    ## Return the corresponding condition to connect tool to particles of the nearest surface
    if plane_name == "max_y":
        surface_mask_receive = (s_receiv[:,:,1] >= max_y)
        surface_mask_send = (s_sender[:,:,1] >= max_y)
    elif plane_name == "max_x":
        surface_mask_receive = (s_receiv[:,:,0] >= max_x)
        surface_mask_send = (s_sender[:,:,0] >= max_x)
    elif plane_name == "max_z":
        surface_mask_receive = (s_receiv[:,:,2] >= max_z)
        surface_mask_send = (s_sender[:,:,2] >= max_z)
    elif plane_name == "min_x":
        surface_mask_receive = (s_receiv[:,:,0] <= min_x)
        surface_mask_send = (s_sender[:,:,0] <= min_x)
    elif plane_name == "min_z":
        surface_mask_receive = (s_receiv[:,:,2] <= min_z)
        surface_mask_send = (s_sender[:,:,2] <= min_z)
    else:
        raise Exception("Unknown plane for connecting tool to surface object particles!!")
    return surface_mask_receive, surface_mask_send

def construct_edges_from_states(states, adj_thresh, mask, tool_mask, topk=10, connect_tools_all=False, 
                                max_y=None, min_y = None, max_x=None, max_z=None, min_x=None, min_z=None,
                                connect_tools_surface=False, connect_tool_all_non_fixed=True, kNN=1.0):
    # :param states: (N, state_dim) torch tensor
    # :param adj_thresh: float
    # :param mask: (N) torch tensor, true when index is a valid particle
    # :param tool_mask: (N) torch tensor, true when index is a valid tool particle
    # :return:
    # - Rr: (n_rel, N) torch tensor
    # - Rs: (n_rel, N) torch tensor
    # N = max_nobj + n_eef

    N, state_dim = states.shape
    s_receiv = states[:, None, :].repeat(1, N, 1) #(N, N, state_dim)
    s_sender = states[None, :, :].repeat(N, 1, 1)

    # dis: particle_num x particle_num
    # adj_matrix: particle_num x particle_num
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # (N, N, state_dim)
    dis = torch.sum(s_diff ** 2, -1)
    mask_1 = mask[:, None].repeat(1, N)
    mask_2 = mask[None, :].repeat(N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    tool_mask_1 = tool_mask[:, None].repeat(1, N)
    tool_mask_2 = tool_mask[None, :].repeat(N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations

    obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
    obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender
        
    adj_matrix = ((dis - threshold) < 0).float()
    # print(f"shape of adjacency matrix: {adj_matrix.size()}")
    # print(f"unique elems in adj: {torch.unique(adj_matrix)}")
    # print(f"shape of dis: {dis.size()}")
    # print(f"shape of s_receiv: {s_receiv.size()}")
    # print(f"s_receiv: \n{s_receiv}")

    # add topk constraints
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix
    print(f"after topk shape of adjacency matrix: {adj_matrix.size()}, dtype: {adj_matrix.dtype}")
    print(f"topk matrix size: {topk_matrix.size()}, sum: {torch.sum(topk_matrix)}, dtype: {topk_matrix.dtype}")
    print(f"dis dtype: {dis.dtype}")
    # print(f"after topk unique elems in adj: {torch.unique(adj_matrix)}")

    if connect_tools_all:
        adj_matrix[obj_tool_mask_1] = 0
        adj_matrix[obj_tool_mask_2] = 1
        adj_matrix[tool_mask_12] = 0  # avoid tool to tool relations
        # print(f"after connect_tools_all: \n{adj_matrix}")

    if connect_tool_all_non_fixed and max_y is not None and min_y is not None:
        ## Connect tool to all non-fixed object particles
        ## Determine if there are any adjacent points for particle receiver, tool sender
        adj_tool_sender = adj_matrix[obj_tool_mask_2.to("cpu")]
        check = torch.sum(adj_tool_sender)
        print(f"determine if there are any adjacent points check: {check}")
        print(f"adj_tool_sender dtype: {adj_tool_sender.dtype}")
        print(f"max_y: {max_y}, min_y: {min_y}")
        ## Bottom 10% of y-coordinates are fixed particles
        threshold = (max_y - min_y) * 0.1 + min_y
        if check > 0:
            dev = mask_1.get_device()
            if dev >= 0:
                surface_mask_receive = (s_receiv[:,:,1] > threshold).to(dev) * mask_1
                surface_mask_send = (s_sender[:,:,1] > threshold).to(dev) * mask_2
            else:
                surface_mask_receive = (s_receiv[:,:,1] > threshold) * mask_1
                surface_mask_send = (s_sender[:,:,1] > threshold) * mask_2
            surf_obj_tool_mask_1 = tool_mask_1 * surface_mask_send  # particle sender, tool receiver
            surf_obj_tool_mask_2 = tool_mask_2 * surface_mask_receive  # particle receiver, tool sender
            print(f"obj_tool_mask1 shape: {obj_tool_mask_1.size()}")
            print(f"obj_tool_mask2 shape: {obj_tool_mask_2.size()}")
            print(f"obj_tool_mask1 true: {torch.sum(obj_tool_mask_1)}")
            print(f"obj_tool_mask2 true: {torch.sum(obj_tool_mask_2)}")
            print(f"surf_obj_tool_mask_1 true: {torch.sum(surf_obj_tool_mask_1)}")
            max_obj_receiv_tool_send_rels = int(torch.sum(surf_obj_tool_mask_2))
            print(f"surf_obj_tool_mask_2 true: {max_obj_receiv_tool_send_rels}")
            adj_matrix[surf_obj_tool_mask_1] = 0
            adj_matrix[surf_obj_tool_mask_2] = 1

            if kNN < 1.0 and kNN > 0.0:
                ## Connect tool to non-fixed object particles within k-Nearest neighbors, by some percent
                # Find the distance of each non-fixed object particle (receiver) to tool (sender)
                # Keep the closest kNN percent of them, sort the distance in ascending order
                keepK = int(kNN * max_obj_receiv_tool_send_rels)
                print(f"--------------------kNN < 1.0 for connect tool to non-fixed obj particles------------------------")
                print(f"Max tool to non-fixed obj particles: {max_obj_receiv_tool_send_rels}, kNN: {kNN}, keepK: {keepK}")
                keepK_idx = torch.topk(dis[surf_obj_tool_mask_2.to("cpu")], k=keepK, dim=-1, largest=False)[1]
                keepK_matrix = torch.zeros_like(dis[surf_obj_tool_mask_2.to("cpu")])
                keepK_matrix.scatter_(-1, keepK_idx, 1)
                print(f"unique elems in keepK matrix: {torch.unique(keepK_matrix)}")
                print(f"Num true in keepK matrix: {torch.sum(keepK_matrix)}")
                print(f"adj matrix type: {adj_matrix.dtype}, keepK matrix type: {keepK_matrix.dtype}")
                print(f"adj matrix device: {adj_matrix.get_device()}, surface indices device: {surf_obj_tool_mask_2.get_device()}, keepK indices dev: {keepK_idx.get_device()}")
                adj_matrix[surf_obj_tool_mask_2.to("cpu")] = adj_matrix[surf_obj_tool_mask_2.to("cpu")] * keepK_matrix.float()
            adj_matrix[tool_mask_12] = 0  # avoid tool to tool relations
            print(f"adj matrix true: {torch.sum(adj_matrix)}")
            # print(f"after connect tool to all non-fixed particles shape of adjacency matrix: {adj_matrix.size()}")
            # print(f"after connect tool to all non-fixed particles shape unique elems in adj: {torch.unique(adj_matrix)}")
    
    if connect_tools_surface and max_y is not None and max_x is not None and min_x is not None and max_z is not None and min_z is not None:
        # Determine closest "surface" based on x, y, or z axis with k-NN, like above
        ## Only attach tool when there is at least one connection between object and tool particles based on distance threshold
        print(f"max_y: {max_y}, min_x: {min_x}, max_x: {max_x}, min_z: {min_z}, max_z: {max_z}")
        ## Determine if there are any adjacent points for particle receiver, tool sender
        adj_tool_sender = adj_matrix[obj_tool_mask_2.to("cpu")]
        check = torch.sum(adj_tool_sender)
        # if dev_idx > 0:
        #     check = torch.sum(adj_matrix.to(dev_idx)[obj_tool_mask_2])
        # else:
        #     check = torch.sum(adj_matrix.to("cpu").detach()[obj_tool_mask_2])
        print(f"check: {check}")
        print(f"adj_tool_sender dtype: {adj_tool_sender.dtype}")
        if check > 0:
            ## First determine which plane the tool is closest to: max_y, min_x, max_x, min_z, or max_z
            ## From the particle points adjacent to tool, determine which plane they are closer to
            print(f"size of s_receive for adj_tool_sender: {s_receiv[adj_tool_sender.long()].size()}")
            dist_min_x = torch.sum((s_receiv[adj_tool_sender.long()][:,:,0] - min_x)**2)
            dist_max_x = torch.sum((s_receiv[adj_tool_sender.long()][:,:,0] - max_x)**2)
            dist_min_z = torch.sum((s_receiv[adj_tool_sender.long()][:,:,2] - min_z)**2)
            dist_max_z = torch.sum((s_receiv[adj_tool_sender.long()][:,:,2] - max_z)**2)
            dist_max_y = torch.sum((s_receiv[adj_tool_sender.long()][:,:,1] - max_y)**2)
            plane_name,plane_val, second_plane_name, second_plane_val = determine_closest_plane(dist_max_y, dist_min_x, dist_max_x, dist_min_z, dist_max_z)
            
            # Determine the closest two surface planes to the tool
            receive_cond_1, send_cond_1 = conditions_for_tool_to_surface(plane_name, max_y, max_x, max_z, min_x, min_z, s_receiv, s_sender)
            receive_cond_2, send_cond_2 = conditions_for_tool_to_surface(second_plane_name, max_y, max_x, max_z, min_x, min_z, s_receiv, s_sender)
            dev = mask_1.get_device()
            if dev >= 0:
                surface_mask_receive = (receive_cond_1 *receive_cond_2).to(dev) * mask_1
                surface_mask_send = (send_cond_1 * send_cond_2).to(dev) * mask_2
            else:
                surface_mask_receive = (receive_cond_1 * receive_cond_2) * mask_1
                surface_mask_send = (send_cond_1 * send_cond_2) * mask_2
            surf_obj_tool_mask_1 = tool_mask_1 * surface_mask_send  # particle sender, tool receiver
            surf_obj_tool_mask_2 = tool_mask_2 * surface_mask_receive  # particle receiver, tool sender
            print(f"obj_tool_mask1 shape: {obj_tool_mask_1.size()}")
            print(f"obj_tool_mask2 shape: {obj_tool_mask_2.size()}")
            print(f"obj_tool_mask1 true: {torch.sum(obj_tool_mask_1)}")
            print(f"obj_tool_mask2 true: {torch.sum(obj_tool_mask_2)}")
            print(f"surf_obj_tool_mask_1 true: {torch.sum(surf_obj_tool_mask_1)}")
            print(f"surf_obj_tool_mask_2 true: {torch.sum(surf_obj_tool_mask_2)}")
            adj_matrix[surf_obj_tool_mask_1] = 0
            adj_matrix[surf_obj_tool_mask_2] = 1
            adj_matrix[tool_mask_12] = 0  # avoid tool to tool relations
            print(f"after connect tool surface shape of adjacency matrix: {adj_matrix.size()}")
            print(f"after connect tool surface unique elems in adj: {torch.unique(adj_matrix)}")

    n_rels = adj_matrix.sum().long().item()
    rels_idx = torch.arange(n_rels).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    print(f"n_rels: {n_rels}, rels shape: {rels.size()}")
    Rr = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rr[rels_idx, rels[:, 0]] = 1
    Rs[rels_idx, rels[:, 1]] = 1
    return Rr, Rs

def construct_edges_from_states_batch(states, adj_thresh, mask, tool_mask, topk=10, connect_tools_all=False):  
    # :param states: (B, N, state_dim) torch tensor
    # :param adj_thresh: (B, ) torch tensor
    # :param mask: (B, N) torch tensor, true when index is a valid particle
    # :param tool_mask: (B, N) torch tensor, true when index is a valid tool particle
    # :return:
    # - Rr: (B, n_rel, N) torch tensor
    # - Rs: (B, n_rel, N) torch tensor

    B, N, state_dim = states.shape
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)

    # dis: B x particle_num x particle_num
    # adj_matrix: B x particle_num x particle_num
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # (N, N, state_dim)
    dis = torch.sum(s_diff ** 2, -1)
    mask_1 = mask[:, :, None].repeat(1, 1, N)
    mask_2 = mask[:, None, :].repeat(1, N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations

    obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
    obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender

    obj_pad_tool_mask_1 = tool_mask_1 * (~tool_mask_2)

    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # add topk constraints
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    if connect_tools_all:
        batch_mask = (adj_matrix[obj_pad_tool_mask_1].reshape(B, -1).sum(-1) > 0)[:, None, None].repeat(1, N, N)
        batch_obj_tool_mask_1 = obj_tool_mask_1 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_1 = obj_tool_mask_1 * (~batch_mask)  # (B, N, N)
        batch_obj_tool_mask_2 = obj_tool_mask_2 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_2 = obj_tool_mask_2 * (~batch_mask)  # (B, N, N)

        adj_matrix[batch_obj_tool_mask_1] = 0
        adj_matrix[batch_obj_tool_mask_2] = 1
        adj_matrix[neg_batch_obj_tool_mask_1] = 0
        adj_matrix[neg_batch_obj_tool_mask_2] = 0

    n_rels = adj_matrix.sum(dim=(1,2))
    n_rel = n_rels.max().long().item()
    rels_idx = []
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    return Rr, Rs
        