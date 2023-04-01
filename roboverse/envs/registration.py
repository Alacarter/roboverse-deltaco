import gym

from roboverse.assets.meta_env_object_lists import (
    PICK_PLACE_TRAIN_TASK_OBJECTS,
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP,
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP
)

ENVIRONMENT_SPECS = (
    {
        'id': 'Widow250Grasp-v0',
        'entry_point': 'roboverse.envs.widow250:Widow250Env',
        'kwargs': {'reward_type': 'grasping',
                   'control_mode': 'discrete_gripper',
                   'target_object': 'beer_bottle',
                   'load_tray': True,
                   'xyz_action_scale': 0.2,
                   }
    },
    {
        'id': 'Widow250PickPlace-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'object_names': ('shed',),
                   'object_scales': (0.7,),
                   'target_object': 'shed',
                   'load_tray': False,
                   'object_position_low': (.49, .18, -.30),
                   'object_position_high': (.59, .27, -.30),
                   'container_name': 'bowl_small',
                   }
    },
    {
        'id': 'Widow250PickPlaceMetaTrainResetFullMultiObjectTwoContainer-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectConcurrentMultiContainerV2Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': 2,

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'fixed_container_position': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'num_tasks': len(PICK_PLACE_TRAIN_TASK_OBJECTS),
                   }
    },
    {
        'id': 'Widow250PickPlaceMetaTrainResetFullMultiObjectTwoContainerRandomDistractor-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectConcurrentMultiContainerV2Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': None,
                   'num_objects_range': (2, 4), # 2 or 3 objects
                   'random_distractor_objs': True,

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'fixed_container_position': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'num_tasks': len(PICK_PLACE_TRAIN_TASK_OBJECTS),
                   }
    },
    {
        'id': 'Widow250PickPlaceMetaTrainResetFullMultiObjectTwoContainerRandomDistractorRandomTrayQuad-v0',
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectConcurrentMultiContainerV2Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': None,
                   'num_objects_range': (2, 4), # 2 or 3 objects
                   'random_distractor_objs': True,

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'fixed_container_position': True,
                   'random_quadrant_cont_obj_positions': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'num_tasks': len(PICK_PLACE_TRAIN_TASK_OBJECTS),
                   }
    },
    {
        'id': 'Widow250PickPlaceMetaTrainResetFullMultiObjectTwoContainerGRFBLRRandomDistractorRandomTrayQuad-v0',
        # RGFB = 6 container directives: Red, Green, Front, Back, Left, Right
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersV2Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': None,
                   'num_objects_range': (2, 4), # 2 or 3 objects
                   'random_distractor_objs': True,

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'extra_container_identifiers': ['front', 'back', 'left', 'right'],
                   'fixed_container_position': True,
                   'random_quadrant_cont_obj_positions': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'num_tasks': 6 * 2 * len(PICK_PLACE_TRAIN_TASK_OBJECTS),
                   }
    },
    {
        'id': 'Widow250PickPlaceFBLRIdenticalMeshColorObjRndDistractorRndTrayQuad-v0',
        # FBLR = 6 container directives: Red, Green, Front, Back, Left, Right
        # 2 identical objects on scene, 1 different object distractor
        # Total: 32 * 6 = 192 tasks
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiIdenticalObjectConcurrentMultiContainerExtraIdentifiersV2Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': None,
                   'num_objects_range': (3, 4), # 3 objects
                   'random_distractor_objs': True,
                   'min_distance_between_objs': 0.04, # we specify objs by their quadrants.
                   'identical_distractor_mode': "same_color_mesh",

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'extra_container_identifiers': ['front', 'back', 'left', 'right'],
                   'fixed_container_position': True,
                   'random_quadrant_cont_obj_positions': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'num_tasks': 6 * 2 * len(PICK_PLACE_TRAIN_TASK_OBJECTS),
                   }
    },
    {
        'id': 'Widow250PickPlaceFBLRIdenticalMeshObjRndDistractorRndTrayQuad-v0',
        # FBLR = 6 container directives: Red, Green, Front, Back, Left, Right
        # 2 identical objects on scene, 1 different object distractor
        # Total: 32 * 6 = 192 tasks
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiIdenticalObjectConcurrentMultiContainerExtraIdentifiersV2Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': None,
                   'num_objects_range': (3, 4), # 3 objects
                   'random_distractor_objs': True,
                   'min_distance_between_objs': 0.07,
                   'identical_distractor_mode': "same_mesh",

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'extra_container_identifiers': ['front', 'back', 'left', 'right'],
                   'fixed_container_position': True,
                   'random_quadrant_cont_obj_positions': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'num_tasks': 6 * 2 * len(PICK_PLACE_TRAIN_TASK_OBJECTS),
                   }
    },
    {
        'id': 'Widow250PickPlaceGRFBLRObjCRndDistractorRndTrayQuad-v0',
        # RGFBLR = 6 container directives: Red, Green, Front, Back, Left, Right
        # ObjC = Color = 8 extra object directives: PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP.keys()
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersV3Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': None,
                   'num_objects_range': (2, 4), # 2 or 3 objects
                   'random_distractor_objs': True,

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'extra_obj_idf_schemes': ["color"],
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'extra_container_identifiers': ['front', 'back', 'left', 'right'],
                   'fixed_container_position': True,
                   'random_quadrant_cont_obj_positions': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'num_tasks': 6 * (2 * (len(PICK_PLACE_TRAIN_TASK_OBJECTS)) + len(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP.keys())),
                   }
    },
    {
        'id': 'Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0',
        # I am shortening the env name so filenames don't run into length errors (>255)
        # RGFBLR = 6 container directives: Red, Green, Front, Back, Left, Right
        # ObjCS = color + shape
        # Color = 8 extra object directives: PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP.keys()
        # Shape = 10 extra object directives: PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP.keys()
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersV3Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': None,
                   'num_objects_range': (3, 4), # 3 objects
                   'random_distractor_objs': True,

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'distractor_obj_hard_mode_prob': 0.6,
                   'extra_obj_idf_schemes': ["color", "shape"],
                   'extra_obj_idf_instr_trailing_tokens': ["colored", "shaped"],
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'extra_container_identifiers': ['front', 'back', 'left', 'right'],
                   'fixed_container_position': True,
                   'random_quadrant_cont_obj_positions': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'all_obj_added_scaling': 0.9,
                   'min_distance_between_objs': 0.065,
                   'num_tasks': 6 * (2 * (len(PICK_PLACE_TRAIN_TASK_OBJECTS))
                                     + len(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP.keys())
                                     + len(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP.keys())),
                   }
    },
    {
        'id': 'Widow250PickPlaceGRFBLRAmbigObjCSRndDistractorRndTrayQuad-v0',
        # Object tasks are not specified completely with language; require looking at demo.
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersAmbigV3Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': None,
                   'num_objects_range': (3, 4), # 3 objects
                   'random_distractor_objs': True,
                   'num_ambig_distractors': 1,
                   'ambig_instructions': True, # This makes language non-ambiguous

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'distractor_obj_hard_mode_prob': 0.0,
                   'extra_obj_idf_schemes': ["color", "shape"],
                   'extra_obj_idf_instr_trailing_tokens': ["colored", "shaped"],
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'extra_container_identifiers': ['front', 'back', 'left', 'right'],
                   'fixed_container_position': True,
                   'random_quadrant_cont_obj_positions': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'all_obj_added_scaling': 0.9,
                   'min_distance_between_objs': 0.065,
                   'num_tasks': 6 * (2 * (len(PICK_PLACE_TRAIN_TASK_OBJECTS))
                                     + len(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP.keys())
                                     + len(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP.keys())),
                   }
    },
    {
        'id': 'Widow250PickPlaceGRFBLRAmbigDebugObjCSRndDistractorRndTrayQuad-v0',
        # Widow250PickPlaceGRFBLRAmbigObjCSRndDistractorRndTrayQuad-v0 but language completely disambiguates
        'entry_point': 'roboverse.envs.widow250_pickplace'
                       ':Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersAmbigV3Env',
        'kwargs': {'reward_type': 'pick_place',
                   'control_mode': 'discrete_gripper',

                   'load_tray': True,
                   'num_objects': None,
                   'num_objects_range': (3, 4), # 3 objects
                   'random_distractor_objs': True,
                   'num_ambig_distractors': 1,
                   'ambig_instructions': False, # This makes language non-ambiguous

                   'possible_objects': PICK_PLACE_TRAIN_TASK_OBJECTS,
                   'distractor_obj_hard_mode_prob': 0.0,
                   'extra_obj_idf_schemes': ["color", "shape"],
                   'extra_obj_idf_instr_trailing_tokens': ["colored", "shaped"],
                   'container_names': ['low_tray_big_half_green', 'low_tray_big_half_red'],
                   'extra_container_identifiers': ['front', 'back', 'left', 'right'],
                   'fixed_container_position': True,
                   'random_quadrant_cont_obj_positions': True,
                   'container_position_z_offset': 0.01,
                   'init_task_idx': 0,
                   'all_obj_added_scaling': 0.9,
                   'min_distance_between_objs': 0.065,
                   'num_tasks': 6 * (2 * (len(PICK_PLACE_TRAIN_TASK_OBJECTS))
                                     + len(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP.keys())
                                     + len(PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP.keys())),
                   }
    },
    # ObjectEnv
    {
        'id': 'PickPlaceTrainObject-v0',
        'entry_point': 'roboverse.envs.objects_env:ObjectsEnv',
        'kwargs': {}
    },
)


def register_environments():
    for env in ENVIRONMENT_SPECS:
        gym.register(**env)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in ENVIRONMENT_SPECS)

    return gym_ids


def make(env_name, *args, **kwargs):
    env = gym.make(env_name, *args, **kwargs)
    return env
