import os
import sys
sys.path.append(".")
sys.path.append("..")
import numpy as np
import pyflex
import gym
import math
from scipy.spatial.distance import cdist

import pybullet as p
import pybullet_data

from sim.sim_env.robot_env import FlexRobotHelper
pyflex.loadURDF = FlexRobotHelper.loadURDF
pyflex.resetJointState = FlexRobotHelper.resetJointState
pyflex.getRobotShapeStates = FlexRobotHelper.getRobotShapeStates

from sim.sim_env.flex_scene import FlexScene
from sim.sim_env.cameras import Camera
from sim.utils import fps_with_idx, quatFromAxisAngle, find_min_distance, rand_float

class FlexEnv(gym.Env):
    def __init__(self, config=None) -> None:
        super().__init__()

        self.dataset_config = config['dataset']
        
        # env component
        self.obj = self.dataset_config['obj']
        self.scene = FlexScene()
        
        # set up pybullet
        physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")

        # set up robot arm
        # xarm6
        self.flex_robot_helper = FlexRobotHelper()
        self.end_idx = self.dataset_config['robot_end_idx']
        self.num_dofs = self.dataset_config['robot_num_dofs']
        self.robot_speed_inv = self.dataset_config['robot_speed_inv']

        # set up pyflex
        self.screenWidth = self.dataset_config['screenWidth']
        self.screenHeight = self.dataset_config['screenHeight']
        self.camera = Camera(self.screenWidth, self.screenHeight)

        pyflex.set_screenWidth(self.screenWidth)
        pyflex.set_screenHeight(self.screenHeight)
        pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
        pyflex.set_light_fov(70.)
        pyflex.init(self.dataset_config['headless'])

        # set up camera
        self.camera_view = self.dataset_config['camera_view']

        # define action space
        self.action_dim = self.dataset_config['action_dim']
        self.action_space = self.dataset_config['action_space']
        
        # stat
        self.count = 0
        self.imgs_list = []
        self.particle_pos_list = []
        self.eef_states_list = []
        self.particle_2_obj_inst_list = []
        self.particle_inv_weight_0 = []
        
        self.fps = self.dataset_config['fps']
        self.fps_number = self.dataset_config['fps_number']
        
        # others
        self.gripper = self.dataset_config['gripper']
        self.stick_len = self.dataset_config['pusher_len']

        # action type
        self.poke = False
    
    ### shape states
    def robot_to_shape_states(self, robot_states):
        n_robot_links = robot_states.shape[0]
        n_table = self.table_shape_states.shape[0]
        
        shape_states = np.zeros((n_table + n_robot_links, 14))
        shape_states[:n_table] = self.table_shape_states # set shape states for table
        shape_states[n_table:] = robot_states # set shape states for robot
        
        return shape_states
                        
    def reset_robot(self, jointPositions = np.zeros(13).tolist()):  
        index = 0
        for j in range(7):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robotId, j)

            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                pyflex.resetJointState(self.flex_robot_helper, j, jointPositions[index])
                index = index + 1
                
        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))
    
    def add_table(self):
        ## add table board
        self.table_shape_states = np.zeros((2, 14))
        # table for workspace
        self.wkspace_height = 0.5
        self.wkspace_width = 3.5 # 3.5*2=7 grid = 700mm
        self.wkspace_length = 4.5 # 4.5*2=9 grid = 900mm
        halfEdge = np.array([self.wkspace_width, self.wkspace_height, self.wkspace_length])
        center = np.array([0.0, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.table_shape_states[0] = np.concatenate([center, center, quats, quats])
        
        # table for robot
        if self.obj in ['cloth', 'bunnybath', 'multiobj']: #'softbody'
            robot_table_height = 0.5 + 1.0
        else:
            robot_table_height = 0.5 + 0.3
        robot_table_width = 126 / 200 # 126mm
        robot_table_length = 126 / 200 # 126mm
        halfEdge = np.array([robot_table_width, robot_table_height, robot_table_length])
        center = np.array([-self.wkspace_width-robot_table_width, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (160. / 255.)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.table_shape_states[1] = np.concatenate([center, center, quats, quats])
    
    def add_empty_box(self):
        ## Add an empty box on the table for packing scene
        # TODO: add 3 walls to the box
        self.box_height = 0.25
        self.box_width = 1.0 # 1.0*2=2 grid = 200mm
        self.box_length = 2.0 # 2.0*2=4 grid = 400mm
        halfEdge = np.array([self.box_width, self.box_height, self.box_length])
        center = np.array([0.0, 0.0, (self.wkspace_length-self.box_length)/2.0])
        quats = quatFromAxisAngle(axis=np.array([0., 1., 0.]), angle=0.)
        hideShape = 0
        color = np.ones(3) * (50. / 255.)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.empty_box_shape_states = np.concatenate([center, center, quats, quats])
    
    def add_robot(self):
        if self.obj in ['granular', 'softbody']:
            # flat board pusher
            robot_base_pos = [-self.wkspace_width-0.6, 0., self.wkspace_height+0.3]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'sim/assets/xarm/xarm6_with_gripper_board.urdf', 
                                           robot_base_pos, robot_base_orn, globalScaling=10.0) 
            self.rest_joints = np.zeros(8)
        elif self.obj in ['rope']:
            # stick pusher
            robot_base_pos = [-self.wkspace_width-0.6, 0., self.wkspace_height+0.3]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'sim/assets/xarm/xarm6_with_gripper.urdf', 
                                           robot_base_pos, robot_base_orn, globalScaling=10.0) 
            self.rest_joints = np.zeros(8)
        elif self.obj in ['cloth', 'bunnybath', 'multiobj']: #'softbody'
            # gripper
            robot_base_pos = [-self.wkspace_width-0.6, 0., self.wkspace_height+1.0]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(self.flex_robot_helper, 'sim/assets/xarm/xarm6_with_gripper_grasp.urdf', 
                                            robot_base_pos, robot_base_orn, globalScaling=10.0) 
            self.rest_joints = np.zeros(13)
    
    def store_data(self, store_cam_param=False, init_fps=False):
        saved_particles = False
        img_list = []
        for j in range(len(self.camPos_list)):
            pyflex.set_camPos(self.camPos_list[j])
            pyflex.set_camAngle(self.camAngle_list[j])
            
            if store_cam_param:
                self.cam_intrinsic_params[j], self.cam_extrinsic_matrix[j] = self.camera.get_cam_params()
            
            # save images
            img = self.render()
            img_list.append(img)

            # Num shapes and rigids
            # print(f"number of shapes: {pyflex.get_n_shapes()}")
            # print(f"number of rigids: {pyflex.get_n_rigids()}")
            # print(f"get mesh edges shape: {pyflex.get_edges().shape}")
            # if pyflex.get_edges().shape[0] > 0:
            #     print(pyflex.get_edges()[0])
            # print(f"get mesh faces shape: {pyflex.get_faces().shape}")
            # if pyflex.get_faces().shape[0] > 0:
            #     print(pyflex.get_faces()[0])
            
            # save particles
            if not saved_particles:
                # save particle pos
                particles = self.get_positions().reshape(-1, 4)
                particles_pos = particles[:, :3]
                # Boolean mask of which particles are fixed (inv weight is 0)
                # TODO: make this a sparser representation by just storing indices of fixed particles
                particles_inv_weights = particles[:, 3]
                particles_inv_weights = (particles_inv_weights == 0.0)
                # Mapping of particle to object instance
                part_2_obj = self.get_part_2_instance()
                # print(f"particle positions shape: {particles.shape}")
                # print(f"part2obj shape: {part_2_obj.shape}")
                # print(f"unique values in the part2obj array: {np.unique(part_2_obj)}")
                # print(part_2_obj)
                if self.fps:
                    if init_fps:
                        _, self.sampled_idx = fps_with_idx(particles_pos, self.fps_number)
                    particles_pos = particles_pos[self.sampled_idx]
                    part_2_obj = part_2_obj[self.sampled_idx]
                    particles_inv_weights = particles_inv_weights[self.sampled_idx]
                self.particle_pos_list.append(particles_pos)
                self.particle_2_obj_inst_list.append(part_2_obj)
                self.particle_inv_weight_0.append(particles_inv_weights)
                # save eef pos
                robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                if self.gripper:
                    eef_states = np.zeros((2, 14))
                    eef_states[0] = robot_shape_states[9] # left finger
                    eef_states[1] = robot_shape_states[12] # right finger
                else:
                    eef_states = np.zeros((1, 14))
                    eef_states[0] = robot_shape_states[-1] # pusher
                self.eef_states_list.append(eef_states)
                
                saved_particles = True
        
        img_list_np = np.array(img_list)
        self.imgs_list.append(img_list_np)
        self.count += 1
    
    ### setup gripper
    def _set_pos(self, picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3] #picker_pos
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)
    
    def _reset_pos(self, particle_pos):
        pyflex.set_positions(particle_pos)
    
    def robot_close_gripper(self, close, jointPoses=None):
        for j in range(8, self.num_joints):
            pyflex.resetJointState(self.flex_robot_helper, j, close)
        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))            
    
    def robot_open_gripper(self):
        for j in range(8, self.num_joints):
            pyflex.resetJointState(self.flex_robot_helper, j, 0.0)
        
    ### reset env
    def reset(self, save_data=False):
        ## init sim env
        # set scene
        self.scene.set_scene(self.obj)
        # set camera
        self.camera.set_init_camera(self.camera_view)
        if save_data:
            self.camPos_list, self.camAngle_list, self.cam_intrinsic_params, self.cam_extrinsic_matrix \
                = self.camera.init_multiview_cameras()
        # add table
        self.add_table()
        # add empty box for packing
        # TODO: make the box hollow
        #self.add_empty_box()
        ## add robot
        self.add_robot()
        pyflex.set_shape_states(self.robot_to_shape_states(pyflex.getRobotShapeStates(self.flex_robot_helper)))
        
        ## update robot shape states
        for idx, joint in enumerate(self.rest_joints):
            pyflex.set_shape_states(self.robot_to_shape_states(pyflex.resetJointState(self.flex_robot_helper, idx, joint)))
        
        self.num_joints = p.getNumJoints(self.robotId)
        self.joints_lower = np.zeros(self.num_dofs)
        self.joints_upper = np.zeros(self.num_dofs)
        print(f"num joints: {self.num_joints}, num dofs: {self.num_dofs}")
        dof_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotId, i)
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                self.joints_lower[dof_idx] = info[8]
                self.joints_upper[dof_idx] = info[9]
                dof_idx += 1
        self.reset_robot()
        
        # initial render
        for _ in range(200):
            pyflex.step()
        
        # save initial rendering
        if save_data:
            self.store_data(store_cam_param=True, init_fps=True)
        
        # output
        out_data = self.imgs_list, self.particle_pos_list, self.eef_states_list, self.particle_2_obj_inst_list, self.particle_inv_weight_0
        
        return out_data
        
    def step(self, action, save_data=False, data=None):
        """
        action: [start_x, start_z, end_x, end_z]
        for top down poking action: [start_x, start_z, start_y, end_x, end_z, end_y]
        """
        self.count = 0
        self.imgs_list, self.particle_pos_list, self.eef_states_list, self.particle_2_obj_inst_list, self.particle_inv_weight_0 = data
        
        # set up action
        h = 0.5 + self.stick_len
        # print("action shape: ", action.shape)
        # print(f"stick len: {self.stick_len}, h: {h}")
        if (action.shape[0] == 4):
            s_2d = np.concatenate([action[:2], [h]])
            e_2d = np.concatenate([action[2:], [h]])
        else:
            # The actual end effector point is at the top of the stick pusher, not the bottom of the stick
            print(f"og s_2d: {action[:3]}, og e_2d: {action[3:]}")
            s_2d = action[:3] + np.array([0., 0., self.stick_len])
            e_2d = action[3:] + np.array([0., 0., self.stick_len])
        print(f"after adding {self.stick_len}, s_2d: {s_2d}, e_2d: {e_2d}")
        # pusher angle depending on x-axis
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi/2
        else:
            pusher_angle = np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])
        # robot orientation
        orn = np.array([0.0, np.pi, pusher_angle + np.pi/2])
        print(f"pusher_angle: {pusher_angle}")

        # create way points
        if self.gripper:
            way_points = [s_2d + [0., 0., 0.5], s_2d, s_2d, e_2d + [0., 0., 0.5], e_2d]
        else:
            if not self.poke:
                way_points = [s_2d + [0., 0., 0.2], s_2d, e_2d, e_2d + [0., 0., 0.2]]
            else:
                # need stay at the target for a while
                y_increment = (e_2d[2] - s_2d[2]) / 2
                x_increment = (e_2d[0] - s_2d[0]) / 2
                z_increment = (e_2d[1] - s_2d[1]) / 2
                print(f"using new poke waypoints, increment: {y_increment}")
                #way_points = [s_2d, s_2d + [x_increment, z_increment, y_increment], e_2d, e_2d]
                way_points = [s_2d, s_2d + [x_increment, z_increment, y_increment], e_2d, e_2d, e_2d + [-x_increment, -z_increment, -y_increment], s_2d]
        self.reset_robot(self.rest_joints)
        print(way_points)

        # set robot speed
        speed = 1.0/self.robot_speed_inv
        
        # step
        for i_p in range(len(way_points)-1):
            s = way_points[i_p]
            e = way_points[i_p+1]
            steps = int(np.linalg.norm(e-s)/speed) + 1
            print(f"at waypoint: {i_p}, has {steps} steps")
            
            for i in range(steps):
                end_effector_pos = s + (e-s) * i / steps # expected eef position
                end_effector_orn = p.getQuaternionFromEuler(orn)
                jointPoses = p.calculateInverseKinematics(self.robotId, 
                                                        self.end_idx, 
                                                        end_effector_pos, 
                                                        end_effector_orn, 
                                                        self.joints_lower.tolist(), 
                                                        self.joints_upper.tolist(),
                                                        (self.joints_upper - self.joints_lower).tolist(),
                                                        self.rest_joints)
                # print('jointPoses:', jointPoses)
                self.reset_robot(jointPoses)
                pyflex.step()
                
                ## ================================================================
                ## gripper control 
                # TODO: see if it's possible to modify the grasping action to hold whole object
                if self.gripper and i_p >= 1:
                    grasp_thresd = 0.1 
                    obj_pos = self.get_positions().reshape(-1, 4)[:, :3]
                    new_particle_pos = self.get_positions().reshape(-1, 4).copy()
                    
                    ### grasping 
                    if i_p == 1:
                        close = 0
                        start = 0
                        end = 0.7
                        close_steps = 50 #500
                        finger_y = 0.5
                        for j in range(close_steps):
                            robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper) # 9: left finger; 12: right finger
                            left_finger_pos, right_finger_pos = robot_shape_states[9][:3], robot_shape_states[12][:3]
                            left_finger_pos[1], right_finger_pos[1] = left_finger_pos[1] - finger_y, right_finger_pos[1] - finger_y 
                            new_finger_pos = (left_finger_pos + right_finger_pos) / 2
                            
                            if j == 0:
                                # fine the k pick point
                                pick_k = 5
                                left_min_dist, left_pick_index = find_min_distance(left_finger_pos, obj_pos, pick_k)
                                right_min_dist, right_pick_index = find_min_distance(right_finger_pos, obj_pos, pick_k)
                                min_dist, pick_index = find_min_distance(new_finger_pos, obj_pos, pick_k)
                                # save the original setting for restoring
                                pick_origin = new_particle_pos[pick_index]
                            
                            if left_min_dist <= grasp_thresd or right_min_dist <= grasp_thresd:
                                    new_particle_pos[left_pick_index, :3] = left_finger_pos
                                    new_particle_pos[left_pick_index, 3] = 0
                                    new_particle_pos[right_pick_index, :3] = right_finger_pos
                                    new_particle_pos[right_pick_index, 3] = 0
                            self._set_pos(new_finger_pos, new_particle_pos)
                            
                            # close the gripper
                            close += (end - start) / close_steps
                            self.robot_close_gripper(close)
                            pyflex.step()
                    
                    # find finger positions
                    robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper) # 9: left finger; 12: right finger
                    left_finger_pos, right_finger_pos = robot_shape_states[9][:3], robot_shape_states[12][:3]
                    left_finger_pos[1], right_finger_pos[1] = left_finger_pos[1] - finger_y, right_finger_pos[1] - finger_y
                    new_finger_pos = (left_finger_pos + right_finger_pos) / 2
                    # connect pick pick point to the finger
                    new_particle_pos[pick_index, :3] = new_finger_pos
                    new_particle_pos[pick_index, 3] = 0
                    self._set_pos(new_finger_pos, new_particle_pos)
                    
                    self.reset_robot(jointPoses)
                    pyflex.step()
                
                ## ================================================================

                # save img in each step
                if self.poke:
                    obj_pos = self.get_positions().reshape(-1, 4)[:, [0, 1, 2]]
                    obj_pos[:, 2] *= -1
                    # to properly compare with obj_pos, need to reorder the end_effector_pos
                    robot_obj_dist = np.min(cdist(end_effector_pos[:3].reshape(1, 3), obj_pos))
                else:
                    obj_pos = self.get_positions().reshape(-1, 4)[:, [0, 2]]
                    obj_pos[:, 1] *= -1
                    robot_obj_dist = np.min(cdist(end_effector_pos[:2].reshape(1, 2), obj_pos))
                if save_data:
                    rob_obj_dist_thresh = self.dataset_config['rob_obj_dist_thresh']
                    contact_interval = self.dataset_config['contact_interval']
                    non_contact_interval = self.dataset_config['non_contact_interval']
                    if robot_obj_dist < rob_obj_dist_thresh and i % contact_interval == 0: # robot-object contact
                        print(f"robot contact with object!!")
                        self.store_data()
                    elif i % non_contact_interval == 0: # not contact
                        self.store_data()
                    
                self.reset_robot()
                if math.isnan(self.get_positions().reshape(-1, 4)[:, 0].max()):
                    print('simulator exploded when action is', action)
                    return None
        
        # set up gripper
        if self.gripper:
            self.robot_open_gripper()
            # reset the mass for the pick points
            new_particle_pos[pick_index, 3] = pick_origin[:, 3]
            self._reset_pos(new_particle_pos)
        
        self.reset_robot()
        
        for i in range(200):
            pyflex.step()
        
        # save final rendering
        if save_data:
            self.store_data()
        
        obs = self.render()
        out_data = self.imgs_list, self.particle_pos_list, self.eef_states_list, self.particle_2_obj_inst_list, self.particle_inv_weight_0
        
        return obs, out_data
    
    def render(self, no_return=False):
        pyflex.step()
        if no_return:
            return
        else:
            return pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5)
    
    def close(self):
        pyflex.clean()
    
    def sample_action(self, init=False, boundary_points=None, boundary=None):
        if self.obj in ['rope', 'granular']:
            action = self.sample_deform_actions()
            return action
        elif self.obj in ['cloth', 'bunnybath']: #'softbody'
            action, boundary_points, boundary = self.sample_grasp_actions_corner(init, boundary_points, boundary)
            return action, boundary_points, boundary
        elif self.obj in ['softbody']:
            # This movement is in the y-coordinate, x and z should be fixed for each action
            print("^^^^^^^^sampling top down poke action^^^^^^^^^")
            self.poke = True
            action = self.sample_top_down_deform_actions()
            return action
        elif self.obj in ['multiobj']:
            #print("!!!!!!!!!!!!!!!!!grasping whole obj!!!!!!!!!!!!!!!")
            #action, boundary_points, boundary = self.sample_grasp_actions_whole_obj(init, boundary_points, boundary)
            action, boundary_points, boundary = self.sample_grasp_actions_corner(init, boundary_points, boundary)
            return action, boundary_points, boundary
        else:
            raise ValueError('action not defined')
    
    def sample_deform_actions(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1 # align with the coordinates
        num_points = positions.shape[0]
        pos_xz = positions[:, [0, 2]]
        
        pos_x, pos_z = positions[:, 0], positions[:, 2]
        center_x, center_z = np.median(pos_x), np.median(pos_z)
        chosen_points = []
        for idx, (x, z) in enumerate(zip(pos_x, pos_z)):
            if np.sqrt((x-center_x)**2 + (z-center_z)**2) < 2.0:
                chosen_points.append(idx)
        print(f'chosen points {len(chosen_points)} out of {num_points}.')
        if len(chosen_points) == 0:
            print('no chosen points')
            chosen_points = np.arange(num_points)
        
        # random choose a start point which can not be overlapped with the object
        valid = False
        for _ in range(1000):
            startpoint_pos_origin = np.random.uniform(-self.action_space, self.action_space, size=(1, 2))
            startpoint_pos = startpoint_pos_origin.copy()
            startpoint_pos = startpoint_pos.reshape(-1)

            # choose end points which is the expolation of the start point and obj point
            pickpoint = np.random.choice(chosen_points)
            obj_pos = positions[pickpoint, [0, 2]]
            slope = (obj_pos[1] - startpoint_pos[1]) / (obj_pos[0] - startpoint_pos[0])
            if obj_pos[0] < startpoint_pos[0]:
                # 1.0 for planning
                # (1.5, 2.0) for data collection
                x_end = obj_pos[0] - 1.0 #rand_float(1.5, 2.0)
            else:
                x_end = obj_pos[0] + 1.0 #rand_float(1.5, 2.0)
            y_end = slope * (x_end - startpoint_pos[0]) + startpoint_pos[1]
            endpoint_pos = np.array([x_end, y_end])
            if obj_pos[0] != startpoint_pos[0] and np.abs(x_end) < 1.5 and np.abs(y_end) < 1.5 \
                and np.min(cdist(startpoint_pos_origin, pos_xz)) > 0.2:
                valid = True
                break
        
        if valid:
            action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        else:
            action = None
        
        return action
    
    def sample_top_down_deform_actions(self):
        ## Have robot poke at the object from top down
        ## Returns a 6-dim vector [x_start, z_start, y_start, x_end, z_end, y_end]
        # Choose an x,z coordinate (need to slightly change x or z coordinate between start/end), change the y coordinate
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1 # align with the coordinates
        num_points = positions.shape[0]
        # pos_xz = positions[:, [0, 2]]
        pos_xyz = positions[:, [0,1,2]]
        # print(f"size of pos_xz: {pos_xz.shape}, size of pos_xyz: {pos_xyz.shape}")
        
        pos_x, pos_y, pos_z = positions[:, 0], positions[:, 1], positions[:, 2]
        center_x, center_y, center_z = np.median(pos_x), np.median(pos_y), np.median(pos_z)
        max_y = np.max(pos_y)
        chosen_points = []
        # no chance to do a surface poke
        is_surface_poke = False #(np.random.uniform(0.0, 1.0) <= 0.1)
        print(f"choosing a surface-level particle: {is_surface_poke}")
        for idx, (x, y, z) in enumerate(zip(pos_x, pos_y, pos_z)):
            # only choose obj particles that are upper 3rd quadrant of y_coordinates
            # choose obj particles that are above the table
            # sample surface particles only instead of middle ones (like top 5% of y coordinates) y >= (max_y * 0.95)
            # sample both surface and some penetration points, upper half
            # end effector point is at the top of the stick, not the bottom, so add self.stick_len back
            if np.sqrt((x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2) < 2.0 and y >= self.wkspace_height:
                # Point cannot be farther down than stick len
                # the distance from particle to top of the object can't be more than the stick len
                dist = np.absolute(y - max_y)
                if dist > self.stick_len:
                    continue
                # choose surface points only or also include middle points
                if is_surface_poke:
                    if y >= (max_y * 0.95):
                        chosen_points.append(idx)
                else:
                    # any point in the object
                    # want medium to deeper pokes
                    #if y >= center_y:
                    if y < (0.9 * max_y):
                        chosen_points.append(idx)
        print(f'chosen points {len(chosen_points)} out of {num_points}.')
        if len(chosen_points) == 0:
            print('no chosen points')
            chosen_points = np.arange(num_points)
        
        # random choose a start point for x,z within the object bounds (and slightly disturb x by 1.0)
        # y needs to be above the object
        # print(f"action dim for top down poking: {self.action_dim}")
        # print(f"action space for top down poking: {self.action_space}")
        valid = False
        for _ in range(1000):
            # choose end points which is the expolation of the start point and obj point
            pickpoint = np.random.choice(chosen_points)
            obj_pos = positions[pickpoint, :]

            # startpoint_pos = [x_start, z_start, y_start]
            #startpoint_pos_origin = np.random.uniform(-self.action_space, self.action_space, size=(1, 3))
            #y_start = np.random.uniform(np.max(pos_y) + 2.0, np.max(pos_y) + 5.0)
            x_disturb = np.random.uniform(-0.5, 0.5)
            z_disturb = np.random.uniform(-0.5, 0.5)
            #if (np.abs(x_disturb) > 0.5):
                # larger x disturb --> higher y start
            #    y_start = np.random.uniform(max_y + 0.5, max_y + 3.0)
            #else:
            y_start = np.random.uniform(max_y + 0.5, max_y + 2.0)
            startpoint_pos_origin = np.array([obj_pos[0]+x_disturb, obj_pos[2]+z_disturb, y_start]).reshape(1,3)
            startpoint_pos = startpoint_pos_origin.copy()
            startpoint_pos = startpoint_pos.reshape(-1)
            # similar x,z as the target obj particle pos
            #slope = (obj_pos[1] - startpoint_pos[1]) / (obj_pos[0] - startpoint_pos[0])
            vertical = (obj_pos[1] - startpoint_pos[2])
            # if obj_pos[0] < startpoint_pos[0]:
            #     # 1.0 for planning
            #     # (1.5, 2.0) for data collection
            #     x_end = obj_pos[0] - 1.0 #rand_float(1.5, 2.0)
            # else:
            #     x_end = obj_pos[0] + 1.0 #rand_float(1.5, 2.0)
            #y_end = slope * (x_end - startpoint_pos[0]) + startpoint_pos[1]
            y_end = obj_pos[1] # go a bit beyond the point to poke it
            # endpoint is particle's position
            endpoint_pos = np.array([obj_pos[0], obj_pos[2], y_end]) #np.array([x_end, y_end])
            #and np.abs(x_end) < 1.5 and np.abs(y_end) < 1.5
            if obj_pos[1] != startpoint_pos[2] and vertical < 0 \
                and np.min(cdist(startpoint_pos_origin, pos_xyz)) > 0.2:
                # Need to ensure vertical difference is negative for top down poke
                print(f"sampled valid action in top down poking...")
                valid = True
                break
        
        if valid:
            action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        else:
            action = None
        # action = [x_start, z_start, y_start, x_end, z_end, y_end]
        return action
    
    def sample_grasp_actions_whole_obj(self, init=False, boundary_points=None, boundary=None):
        # TODO: Grasp the width of the object
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1
        particle_x, particle_y, particle_z = positions[:, 0], positions[:, 1], positions[:, 2]
        x_min, y_min, z_min = np.min(particle_x), np.min(particle_y), np.min(particle_z)
        x_max, y_max, z_max = np.max(particle_x), np.max(particle_y), np.max(particle_z)

        # choose the starting point at the boundary of the object
        if init: # record boundary points
            boundary_points = []
            boundary = []
            for idx, point in enumerate(positions):
                if point[0] == x_max:
                    boundary_points.append(idx)
                    boundary.append(1)
                elif point[0] == x_min:
                    boundary_points.append(idx)
                    boundary.append(2)
                elif point[2] == z_max:
                    boundary_points.append(idx)
                    boundary.append(3)
                elif point[2] == z_min:
                    boundary_points.append(idx)
                    boundary.append(4)
        assert len(boundary_points) == len(boundary)

        # random pick a point as start point along x-axis or z-axis
        valid = False
        for _ in range(1000):
            pick_idx = np.random.choice(len(boundary_points))
            if boundary[pick_idx] == 1 or boundary[pick_idx] == 2:
                axis = "x"
                # set z position to be z_min
                startpoint_pos = np.array([positions[boundary_points[pick_idx], 0], z_min])
            else:
                axis = "z"
                # set x position to be x_min
                startpoint_pos = np.array([x_min, positions[boundary_points[pick_idx], 2]])
            # start point is at boundary of one axis and min of the other axis
            #startpoint_pos = positions[boundary_points[pick_idx], [0, 2]]
            endpoint_pos = startpoint_pos.copy()
            # choose end points which is outside the obj
            #move_distance = rand_float(1.0, 1.5)
            if axis == "x":
                # move distance is along the width of z-axis
                move_distance = z_max - z_min
            else:
                # move distance is along the width of x-axis
                move_distance = x_max - x_min
            
            if boundary[pick_idx] == 1:
                #endpoint_pos[0] += move_distance 
                # start point is on x_max
                endpoint_pos[1] = z_max
            elif boundary[pick_idx] == 2:
                # start point is on x_min
                #endpoint_pos[0] -= move_distance
                endpoint_pos[1] = z_max
            elif boundary[pick_idx] == 3:
                # start point at z_max
                #endpoint_pos[1] += move_distance
                endpoint_pos[0] = x_max
            elif boundary[pick_idx] == 4:
                # start point at z_min
                #endpoint_pos[1] -= move_distance
                endpoint_pos[0] = x_max
            
            if np.abs(endpoint_pos[0]) < 3.5 and np.abs(endpoint_pos[1]) < 2.5:
                valid = True
                break
        
        if valid:
            action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        else:
            action = None
        
        return action, boundary_points, boundary
    
    def sample_grasp_actions_corner(self, init=False, boundary_points=None, boundary=None):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1
        particle_x, particle_y, particle_z = positions[:, 0], positions[:, 1], positions[:, 2]
        x_min, y_min, z_min = np.min(particle_x), np.min(particle_y), np.min(particle_z)
        x_max, y_max, z_max = np.max(particle_x), np.max(particle_y), np.max(particle_z)
        
        # choose the starting point at the boundary of the object
        if init: # record boundary points
            boundary_points = []
            boundary = []
            for idx, point in enumerate(positions):
                if point[0] == x_max:
                    boundary_points.append(idx)
                    boundary.append(1)
                elif point[0] == x_min:
                    boundary_points.append(idx)
                    boundary.append(2)
                elif point[2] == z_max:
                    boundary_points.append(idx)
                    boundary.append(3)
                elif point[2] == z_min:
                    boundary_points.append(idx)
                    boundary.append(4)
        assert len(boundary_points) == len(boundary)
        
        # random pick a point as start point
        valid = False
        for _ in range(1000):
            pick_idx = np.random.choice(len(boundary_points))
            startpoint_pos = positions[boundary_points[pick_idx], [0, 2]]
            endpoint_pos = startpoint_pos.copy()
            # choose end points which is outside the obj
            move_distance = rand_float(1.0, 1.5)
            
            if boundary[pick_idx] == 1:
                endpoint_pos[0] += move_distance 
            elif boundary[pick_idx] == 2:
                endpoint_pos[0] -= move_distance
            elif boundary[pick_idx] == 3:
                endpoint_pos[1] += move_distance
            elif boundary[pick_idx] == 4:
                endpoint_pos[1] -= move_distance
            
            if np.abs(endpoint_pos[0]) < 3.5 and np.abs(endpoint_pos[1]) < 2.5:
                valid = True
                break
        
        if valid:
            action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        else:
            action = None
        
        return action, boundary_points, boundary
    
    def get_positions(self):
        return pyflex.get_positions()
    
    def get_part_2_instance(self):
        return pyflex.get_particle_2_obj_instance()
    
    def get_num_particles(self):
        n_particles = int(pyflex.get_n_particles)
        assert n_particles == self.get_positions().reshape(-1, 4).shape[0]
        return self.get_positions().reshape(-1, 4).shape[0]
    
    def get_property_params(self):
        return self.scene.get_property_params()
    
