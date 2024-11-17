import numpy as np
from scipy.spatial.transform import Rotation
from sim.utils import rand_float, quaternion_multuply, rand_int

"""
Support Scenes:
        1. rope_scene
        2. granular_scene
        3. cloth_scene
"""

def rope_scene():
        radius = 0.03
        
        # rope position
        rope_trans = [0., 0.5, 2.0] # [x, y, z]
        
        # rope scale (length and thickness)
        rope_length = rand_float(2.5, 3.0)
        rope_thickness = 3.0
        rope_scale = np.array([rope_length, rope_thickness, rope_thickness]) * 50
        
        # rope stiffness
        stiffness = np.random.rand()
        print(f"rope stiffness for uniform: {stiffness}")
        if stiffness < 0.5:
                global_stiffness = stiffness * 1e-4 / 0.5
                cluster_spacing = 2 + 8 * stiffness
        else:
                global_stiffness = (stiffness - 0.5) * 4e-4 + 1e-4
                cluster_spacing = 6 + 4 * (stiffness - 0.5)
        
        # rope frtction
        dynamicFriction = 0.1
        
        # rope rotation
        z_rotation = rand_float(10, 20) 
        y_rotation = 90. 
        rot_1 = Rotation.from_euler('xyz', [0, y_rotation, 0.], degrees=True)
        rotate_1 = rot_1.as_quat()
        rot_2 = Rotation.from_euler('xyz', [0, 0, z_rotation], degrees=True)
        rotate_2 = rot_2.as_quat()
        rope_rotate = quaternion_multuply(rotate_1, rotate_2)
        
        # others (ususally fixed)
        cluster_radius = 0.
        cluster_stiffness = 0.55

        link_radius = 0. 
        link_stiffness = 1.

        surface_sampling = 0.
        volume_sampling = 4.

        skinning_falloff = 5.
        skinning_max_dist = 100.

        cluster_plastic_threshold = 0.
        cluster_plastic_creep = 0.

        particleFriction = 0.25
        
        draw_mesh = 1

        relaxtion_factor = 1.
        collisionDistance = radius * 0.5
        
        # params
        scene_params = np.array([*rope_scale, *rope_trans, radius, 
                                cluster_spacing, cluster_radius, cluster_stiffness,
                                link_radius, link_stiffness, global_stiffness,
                                surface_sampling, volume_sampling, skinning_falloff, skinning_max_dist,
                                cluster_plastic_threshold, cluster_plastic_creep,
                                dynamicFriction, particleFriction, draw_mesh, relaxtion_factor, 
                                *rope_rotate, collisionDistance])
        
        property_params = {'particle_radius': radius,
                        'length': rope_length,
                        'thickness': rope_thickness,
                        'dynamic_friction': dynamicFriction,
                        'cluster_spacing': cluster_spacing,
                        "global_stiffness": global_stiffness,
                        "stiffness": stiffness,}
        
        return scene_params, property_params
        
def granular_scene():
        radius = 0.03

        granular_scale = rand_float(0.1, 0.3)
        
        area = rand_float(1 ** 2, 3 ** 2) 
        xz_ratio = rand_float(0.8, 1.2)
        x_max = area ** 0.5 * 0.5 * xz_ratio ** 0.5
        x_min = -x_max
        z_max = area ** 0.5 * 0.5 * xz_ratio ** -0.5
        z_min = -z_max
        
        granular_dis = rand_float(0.1 * granular_scale, 0.2 * granular_scale)
        num_granular_ft_x = (x_max - x_min - granular_scale) / (granular_dis + granular_scale) + 1
        num_granular_ft_z = (z_max - z_min - granular_scale) / (granular_dis + granular_scale) + 1            
        
        # shape
        shape_type = 0 # 0: irreular shape; 1: regular shape
        shape_min_dist = 5. # 5. for irregular shape; 8 for regulra shape
        shape_max_dist = 10.
        
        num_granular_ft_y = 1 
        num_granular_ft = [num_granular_ft_x, num_granular_ft_y, num_granular_ft_z] 
        num_granular = int(num_granular_ft_x * num_granular_ft_y * num_granular_ft_z)
        
        pos_granular = [-1., 1., -1.]

        draw_mesh = 1
        
        shapeCollisionMargin = 0.01 
        collisionDistance = 0.03
        
        dynamic_friction = 1.0 
        granular_mass = 0.05 

        scene_params = np.array([radius, *num_granular_ft, granular_scale, *pos_granular, granular_dis, 
                                draw_mesh, shapeCollisionMargin, collisionDistance, dynamic_friction,
                                granular_mass, shape_type, shape_min_dist, shape_max_dist])

        
        property_param = {
        'particle_radius': radius,
        'granular_scale': granular_scale,
        'num_granular': num_granular,
        'distribution_r': granular_dis,
        'dynamic_friction': dynamic_friction,
        'granular_mass': granular_mass,
        'area': area,
        'xz_ratio': xz_ratio,
        }
        
        return scene_params, property_param

def cloth_scene():
        particle_r = 0.03 
        cloth_pos = [-0.5, 1., 0.0]
        cloth_size = np.array([1., 1.]) * 70.
        
        """
        stretch stiffness: resistance to lengthwise stretching
        bend stiffness: resistance to bending
        shear stiffness: resistance to forces that cause sliding or twisting deformation
        """
        sf = np.random.rand()
        stiffness_factor = sf * 1.4 + 0.1
        stiffness = np.array([1.0, 1.0, 1.0]) * stiffness_factor 
        stiffness[0] = np.clip(stiffness[0], 1.0, 1.5)
        dynamicFriction = -sf * 0.9 + 1.0
        
        cloth_mass = 0.1 
        
        render_mode = 2 # 1: particles; 2: mesh
        flip_mesh = 0
        
        staticFriction = 0.0 
        particleFriction = 0.0
        
        scene_params = np.array([*cloth_pos, *cloth_size, *stiffness,
                        cloth_mass, particle_r, render_mode, flip_mesh, 
                        dynamicFriction, staticFriction, particleFriction])
        
        property_params = {'particle_radius': particle_r,
                        'stretch_stiffness': stiffness[0],
                        'bend_stiffness': stiffness[1],
                        'shear_stiffness': stiffness[2],
                        'dynamic_friction': dynamicFriction,
                        'sf': sf,
                        }

        return scene_params, property_params

def softbody_scene():
    """
    https://github.com/jhyau/AdaptiGraph/blob/main/PyFleX/bindings/scenes/by_softbody.h
    Scene params:
        scale: for index [0,1,2]
        trans: for index [3,4,5]
        radius: index 6
        clusterSpacing: index 7
        clusterRadius: index 8
        clusterStiffness: index 9
        linkRadius: index 10
        linkStiffness: index 11
        globalStiffness: index 12
        surfaceSampling: index 13
        volumeSampling: index 14
        skinningFalloff: index 15
        skinningMaxDistance: index 16
        clusterPlasticThreshold: index 17
        clusterPlasticCreep: index 18
        dynamicFriction: index 19
        particleFriction: index 20
        draw_mesh: index 21
        relaxtion factor: index 22
        rotate_v: index [23, 24, 25]
        rotate_w: index 26
        collisionDistance: index 27
    """
    radius = 0.03

    # softbody trans position
    #trans = [0., 0.5, 2.0] # [x, y, z]
    # table's x-axis (width) is shorter, 3.5*2=7 grid = 700mm
    # table's z-axis (length) is longer, 4.5*2=9 grid = 900mm
    trans = [rand_float(0., 1.0), 0.5, rand_float(0., 2.0)]
    print(f"softbody trans: {trans}")
        
    # softbody scale
    #edge_length = rand_float(1.0, 3.0)
    #rope_thickness = 3.0

    # make sure object isn't too large. or else you'd need to modify the number of max particles and the cluster radius
    # otherwise particles will be too spread apart to form edges
    s_scale = rand_int(10, 25)
    # allow greater variance for height, but don't allow anything beyond 80
    scale = np.array([rand_float(2.0, 3.0), rand_float(1.0, 4.0), rand_float(2.0, 3.0)]) * s_scale #* 50
    print(f"softbody scale: {scale} with s_scale: {s_scale}")
    
    # softbody stiffness
    #stiffness = np.random.rand()
    stiffness = 0.1
    print(f"softbody stiffness for uniform/homogeneous: {stiffness}")
    if stiffness < 0.5:
        global_stiffness = stiffness * 1e-4 / 0.5
        cluster_spacing = 2 + 8 * stiffness
    else:
        global_stiffness = (stiffness - 0.5) * 4e-4 + 1e-4
        cluster_spacing = 6 + 4 * (stiffness - 0.5)
    
    # softbody frtction
    dynamicFriction = 0.1
    
    # others (usually fixed)
    cluster_radius = 0.
    cluster_stiffness = 0.55

    link_radius = 0. 
    link_stiffness = 1.

    surface_sampling = 0.
    volume_sampling = 4.

    skinning_falloff = 5.
    skinning_max_dist = 100.

    cluster_plastic_threshold = 0.
    cluster_plastic_creep = 0.

    particleFriction = 0.25
    
    draw_mesh = 0 #1 set 0 for particles only, no mesh

    relaxtion_factor = 1.
    collisionDistance = radius * 0.5

    # ratio of particles (from bottom up) to keep fixed. Set 0 to not have any fixed particles. Usually set to 10 (10%)
    num_fixed_particles = 10

    # coordinate to determine which particles are fixed (x,y, or z)
    fixed_coord = 1

    # Load box (actual cube) [0 or 3] or sphere [1] or cylinder[2]
    # ignore cube mesh (rectangular) for now
    obj_type = 0 #rand_int(0, 4)

    # if cylinder, don't rotate in y direction
    if obj_type == 2:
        # no rotation for cylinder
        #x_rotation = 45.
        z_rotation = 90. #rand_float(10, 20)
        x_rotation = 90. 
        y_rotation = 90.
        rot_1 = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
        rotate = rot_1.as_quat()
        # rot_2 = Rotation.from_euler('xyz', [0, 0, z_rotation], degrees=True)
        # rotate_2 = rot_2.as_quat()
        # rotate = quaternion_multuply(rotate_1, rotate_2)
        # rot_3 = Rotation.from_euler('xyz', [0, y_rotation, 0.], degrees=True)
        # rotate_3 = rot_3.as_quat()
        # rotate = quaternion_multuply(rotate, rotate_3)
    else:
        # softbody rotation
        # Don't rotate the box along x or z axis
        #x_rotation = 45.
        # y axis is height, x is perpendicular to the robot arm, z is along the length of the table
        #z_rotation = rand_float(10, 20)
        y_rotation = rand_float(0., 90.) #90. 
        rot_1 = Rotation.from_euler('xyz', [0, y_rotation, 0.], degrees=True)
        rotate_1 = rot_1.as_quat()
        #rot_2 = Rotation.from_euler('xyz', [0, 0, z_rotation], degrees=True)
        #rotate_2 = rot_2.as_quat()
        #first_rotate = quaternion_multuply(rotate_1, rotate_2)
        #rot_3 = Rotation.from_euler('xyz', [x_rotation, 0, 0], degrees=True)
        #rotate_3 = rot_3.as_quat()
        rotate = rotate_1 #quaternion_multuply(rotate_1, rotate_2)
    
    print("rotate: ", rotate)
    # params
    scene_params = np.array([*scale, *trans, radius, 
                            cluster_spacing, cluster_radius, cluster_stiffness,
                            link_radius, link_stiffness, global_stiffness,
                            surface_sampling, volume_sampling, skinning_falloff, skinning_max_dist,
                            cluster_plastic_threshold, cluster_plastic_creep,
                            dynamicFriction, particleFriction, draw_mesh, relaxtion_factor, 
                            *rotate, collisionDistance, num_fixed_particles, fixed_coord, obj_type])
    
    property_params = {'particle_radius': radius,
                    'cluster_radius': cluster_radius,
                    'dynamic_friction': dynamicFriction,
                    'cluster_spacing': cluster_spacing,
                    "global_stiffness": global_stiffness,
                    "stiffness": stiffness,}
    
    return scene_params, property_params

def yz_bunnybath_scene():
    # By default, the bunnybath.h scene doesn't take in scene params
    # so we use yz_bunnybath.h instead
    # It currently doesn't use the scene_params though, would need to make changes to allow that
    radius = 0.1
    dynamicFriction = 0.01
    viscosity = 2.0

    # deforming bunny params
    s = radius*0.5
    m = 0.25

    cohesion = 0.02
    collisionDistance = 0.01
    scene_params = np.array([0])
    property_params = {'particle_radius': radius,
                    'viscosity': viscosity,
                    'dynamic_friction': dynamicFriction,
                    's': s,
                    'm': m,
                    'cohesion': cohesion,
                    'collisionDistance': collisionDistance,}
    return scene_params, property_params

def multi_obj_scene():
    """
    Scene params:
        scale: for index [0,1,2]
        trans: for index [3,4,5]
        radius: index 6
        clusterSpacing: index 7
        clusterRadius: index 8
        clusterStiffness: index 9
        linkRadius: index 10
        linkStiffness: index 11
        globalStiffness: index 12
        surfaceSampling: index 13
        volumeSampling: index 14
        skinningFalloff: index 15
        skinningMaxDistance: index 16
        clusterPlasticThreshold: index 17
        clusterPlasticCreep: index 18
        dynamicFriction: index 19
        particleFriction: index 20
        draw_mesh: index 21
        relaxtion factor: index 22
        rotate_v: index [23, 24, 25]
        rotate_w: index 26
        collisionDistance: index 27
    """
    radius = 0.03

    # softbody trans position
    trans = [0., 0.5, 2.0] # [x, y, z]
        
    # softbody scale
    edge_length = rand_float(2.5, 3.0)
    print(f"edge_length: {edge_length}")
    #rope_thickness = 3.0
    scale = np.array([edge_length, edge_length, edge_length]) * 10 #* 50
    
    # softbody stiffness
    stiffness = np.random.rand()
    print(f"softbody stiffness for uniform: {stiffness}")
    if stiffness < 0.5:
        global_stiffness = stiffness * 1e-4 / 0.5
        cluster_spacing = 2 + 8 * stiffness
    else:
        global_stiffness = (stiffness - 0.5) * 4e-4 + 1e-4
        cluster_spacing = 6 + 4 * (stiffness - 0.5)
    
    # softbody frtction
    dynamicFriction = 0.1
        
    # softbody rotation
    #x_rotation = 45.
    z_rotation = rand_float(10, 20)
    y_rotation = 90. 
    rot_1 = Rotation.from_euler('xyz', [0, y_rotation, 0.], degrees=True)
    rotate_1 = rot_1.as_quat()
    rot_2 = Rotation.from_euler('xyz', [0, 0, z_rotation], degrees=True)
    rotate_2 = rot_2.as_quat()
    #first_rotate = quaternion_multuply(rotate_1, rotate_2)
    #rot_3 = Rotation.from_euler('xyz', [x_rotation, 0, 0], degrees=True)
    #rotate_3 = rot_3.as_quat()
    rotate = quaternion_multuply(rotate_1, rotate_2)
    
    # others (usually fixed)
    cluster_radius = 0.
    cluster_stiffness = 0.55

    link_radius = 0. 
    link_stiffness = 1.

    surface_sampling = 0.
    volume_sampling = 4.

    skinning_falloff = 5.
    skinning_max_dist = 100.

    cluster_plastic_threshold = 0.
    cluster_plastic_creep = 0.

    particleFriction = 0.25
    
    draw_mesh = 1

    relaxtion_factor = 1.
    collisionDistance = radius * 0.5

    # box scale and trans
    # softbody trans position
    box_trans = [1.0, 0.5, 0.0] # [x, y, z]
    box_edge_length = rand_float(2.5, 3.0)
    print(f"box edge_length: {box_edge_length}")
    #rope_thickness = 3.0
    box_scale = np.array([box_edge_length, box_edge_length, box_edge_length]) * 20

    # flat circle/sphere scale and trans
    # softbody trans position
    flat_trans = [-0.5, 0.5, -1.0] # [x, y, z]

    # softbody scale
    #rope_thickness = 3.0
    axis = rand_float(2.5, 3.0)
    flat_scale = np.array([axis, axis, axis]) * 10 #* 50

    # ratio of particles (from bottom up) to keep fixed
    num_fixed_particles = 0

    # which coordinate to fix (x,y,or z)
    fixed_coord = 1

    # params
    scene_params = np.array([*scale, *trans, radius, 
                            cluster_spacing, cluster_radius, cluster_stiffness,
                            link_radius, link_stiffness, global_stiffness,
                            surface_sampling, volume_sampling, skinning_falloff, skinning_max_dist,
                            cluster_plastic_threshold, cluster_plastic_creep,
                            dynamicFriction, particleFriction, draw_mesh, relaxtion_factor, 
                            *rotate, collisionDistance, *box_scale, *box_trans, *flat_scale, *flat_trans, 
                            num_fixed_particles, fixed_coord])
    
    property_params = {'particle_radius': radius,
                    'cluster_radius': cluster_radius,
                    'dynamic_friction': dynamicFriction,
                    'cluster_spacing': cluster_spacing,
                    "global_stiffness": global_stiffness,
                    "stiffness": stiffness,}
    
    return scene_params, property_params