import pybullet_data
import pybullet as p
import os
from collections import Counter
import importlib.util
import numpy as np
from .control import get_object_position, get_link_state

CUR_PATH = os.path.dirname(os.path.realpath(__file__))
ASSET_PATH = os.path.join(CUR_PATH, '../assets')
SHAPENET_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects/ShapeNetCore')
BASE_ASSET_PATH = os.path.join(ASSET_PATH, 'bullet-objects')
BULLET3_ASSET_PATH = os.path.join(BASE_ASSET_PATH, 'bullet3')

MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS = 500
SHAPENET_SCALE = 0.5


def in_rectangle(object_xy, rect_center_xy, rect_half_extents):
    object_xy = np.array(object_xy)
    rect_center_xy = np.array(rect_center_xy)
    rect_half_extents = np.array(rect_half_extents)
    return (np.all(object_xy < (rect_center_xy + rect_half_extents)) and
            np.all(object_xy > (rect_center_xy - rect_half_extents)))


def check_in_container(object_name,
                       object_id_map,
                       container_pos,
                       place_success_height_threshold,
                       place_success_radius_threshold,
                       container_half_extents=None):
    object_pos, _ = get_object_position(object_id_map[object_name])
    object_height = object_pos[2]
    object_xy = object_pos[:2]
    container_center_xy = container_pos[:2]
    success = False
    if object_height < place_success_height_threshold:
        if container_half_extents is None:
            object_container_distance = np.linalg.norm(
                object_xy - container_center_xy)
            if object_container_distance < place_success_radius_threshold:
                success = True
        else:
            if in_rectangle(
                    object_xy, container_center_xy, container_half_extents):
                success = True

    return success


def check_under_height_threshold(
        object_name, object_id_map, place_success_height_threshold):
    object_pos, _ = get_object_position(object_id_map[object_name])
    object_height = object_pos[2]
    return bool(object_height < place_success_height_threshold)


def check_grasp(object_name,
                object_id_map,
                robot_id,
                end_effector_index,
                grasp_success_height_threshold,
                grasp_success_object_gripper_threshold,
                ):
    object_pos, _ = get_object_position(object_id_map[object_name])
    object_height = object_pos[2]
    success = False
    if object_height > grasp_success_height_threshold:
        ee_pos, _ = get_link_state(
            robot_id, end_effector_index)
        object_gripper_distance = np.linalg.norm(
            object_pos - ee_pos)
        if object_gripper_distance < \
                grasp_success_object_gripper_threshold:
            success = True

    return success


def generate_object_positions_single(
        small_object_position_low, small_object_position_high,
        large_object_position_low, large_object_position_high,
        min_distance_large_obj=0.1):

    valid = False
    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    while not valid:
        large_object_position = np.random.uniform(
                low=large_object_position_low, high=large_object_position_high)
        small_object_positions = []
        small_object_position = np.random.uniform(
            low=small_object_position_low, high=small_object_position_high)
        small_object_positions.append(small_object_position)
        valid = (np.linalg.norm(
            small_object_positions[0] - large_object_position)
            > min_distance_large_obj)
        if i > max_attempts:
            raise ValueError('Min distance could not be assured')

    return large_object_position, small_object_positions


def generate_object_positions_v2(
        small_object_position_low, small_object_position_high,
        large_object_position_low, large_object_position_high,
        min_distance_small_obj=0.07, min_distance_large_obj=0.1):

    valid = False
    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    while not valid:
        large_object_position = np.random.uniform(
                low=large_object_position_low, high=large_object_position_high)
        # large_object_position = np.reshape(large_object_position, (1, 3))

        small_object_positions = []
        for _ in range(2):
            small_object_position = np.random.uniform(
                low=small_object_position_low, high=small_object_position_high)
            small_object_positions.append(small_object_position)

        valid_1 = (
            np.linalg.norm(
                small_object_positions[0] - small_object_positions[1])
            > min_distance_small_obj)
        valid_2 = (
            np.linalg.norm(
                small_object_positions[0] - large_object_position)
            > min_distance_large_obj)
        valid_3 = (
            np.linalg.norm(
                small_object_positions[1] - large_object_position)
            > min_distance_large_obj)

        valid = valid_1 and valid_2 and valid_3
        if i > max_attempts:
            raise ValueError('Min distance could not be assured')

    return large_object_position, small_object_positions


def generate_object_positions_v3(
        small_object_position_low, small_object_position_high,
        large_object_position_low, large_object_position_high,
        target_object, place_success_radius_threshold, reverse=False,
        min_distance_small_obj=0.07, min_distance_large_obj=0.1,
        container_half_extents=None):

    """
    Generates positions with the target object either in or out of the container
    depending on the reverse flag and the non-target object anywhere
    """

    valid = False
    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    while not valid:
        large_object_position = np.random.uniform(
                low=large_object_position_low, high=large_object_position_high)

        small_object_position = np.random.uniform(
            low=small_object_position_low, high=small_object_position_high)
        small_object_position_target = np.random.uniform(
            low=small_object_position_low, high=small_object_position_high)

        object_container_distance = np.linalg.norm(
            small_object_position_target[:2] - large_object_position[:2])
        # circular container
        if container_half_extents is None:
            if reverse:
                valid = (
                    object_container_distance < place_success_radius_threshold)
            else:
                valid = (
                    object_container_distance > place_success_radius_threshold)
        # rectangular container
        else:
            if reverse:
                valid = in_rectangle(
                    small_object_position_target[:2],
                    large_object_position[:2], container_half_extents)
            else:
                valid = not in_rectangle(
                    small_object_position_target[:2],
                    large_object_position[:2], container_half_extents)

        if i > max_attempts:
            raise ValueError('Min distance could not be assured')

    small_object_positions = [
        small_object_position, small_object_position_target]
    return large_object_position, small_object_positions


def generate_object_positions(
        object_position_low, object_position_high,
        num_objects, min_distance=0.07, current_positions=None,
        object_to_range_idx_list=[]):

    def pick_maybe_random_obj_pos_low_high(
            cand_obj_pos_lows, cand_obj_pos_highs, region_idx=None):
        if region_idx is None:
            assert len(cand_obj_pos_lows) == len(cand_obj_pos_highs)
            region_idx = np.random.choice(range(len(cand_obj_pos_lows)))
        else:
            assert type(region_idx) in [int, np.int64]

        object_position_low = cand_obj_pos_lows[region_idx]
        object_position_high = cand_obj_pos_highs[region_idx]
        return object_position_low, object_position_high

    obj_pos_are_lists_of_lists = (
        all([isinstance(elem, list) for elem in object_position_low]) and
        all([isinstance(elem, list) for elem in object_position_high]))

    assert len(object_position_low) == len(object_position_high)
    num_ranges = len(object_position_low)

    if obj_pos_are_lists_of_lists:
        cand_obj_pos_lows = object_position_low
        cand_obj_pos_highs = object_position_high

        if len(object_to_range_idx_list) > 1:
            assert len(object_to_range_idx_list) == num_objects
            assert all([
                (range_idx in range(num_ranges))
                for range_idx in object_to_range_idx_list])
            region_idx = object_to_range_idx_list[0]
        else:
            region_idx = None

        object_position_low, object_position_high = (
            pick_maybe_random_obj_pos_low_high(
                cand_obj_pos_lows, cand_obj_pos_highs, region_idx))

    if current_positions is None:
        object_positions = np.random.uniform(
            low=object_position_low, high=object_position_high)
        object_positions = np.reshape(object_positions, (1, 3))
    else:
        object_positions = current_positions

    max_attempts = MAX_ATTEMPTS_TO_GENERATE_OBJECT_POSITIONS
    i = 0
    while object_positions.shape[0] < num_objects:
        i += 1

        # if obj_position_low/high are list of lists, pick a random bounds
        if obj_pos_are_lists_of_lists:
            if len(object_to_range_idx_list) > 1:
                region_idx = object_to_range_idx_list[object_positions.shape[0]]
            else:
                region_idx = None
            object_position_low, object_position_high = (
                pick_maybe_random_obj_pos_low_high(
                    cand_obj_pos_lows, cand_obj_pos_highs, region_idx))

        object_position_candidate = np.random.uniform(
            low=object_position_low, high=object_position_high)
        object_position_candidate = np.reshape(
            object_position_candidate, (1, 3))
        min_distance_so_far = []
        for o in object_positions:
            dist = np.linalg.norm(o - object_position_candidate)
            min_distance_so_far.append(dist)
        min_distance_so_far = np.array(min_distance_so_far)
        if (min_distance_so_far > min_distance).all():
            object_positions = np.concatenate(
                (object_positions, object_position_candidate), axis=0)

        if i > max_attempts:
            raise ValueError('Min distance could not be assured')

    return object_positions


def import_metadata(asset_path):
    metadata_spec = importlib.util.spec_from_file_location(
        "metadata", os.path.join(asset_path, "metadata.py"))
    metadata = importlib.util.module_from_spec(metadata_spec)
    metadata_spec.loader.exec_module(metadata)
    return metadata.obj_path_map, metadata.path_scaling_map


def import_shapenet_metadata():
    return import_metadata(SHAPENET_ASSET_PATH)


shapenet_obj_path_map, shapenet_path_scaling_map = import_shapenet_metadata()


def load_object(object_name, object_position, object_quat, scale=1.0,
                randomize_object_quat=False, rgba=None):
    if randomize_object_quat:
        from scipy.spatial.transform import Rotation
        object_quat = tuple(Rotation.random().as_quat())

    if object_name in shapenet_obj_path_map.keys():
        obj_id = load_shapenet_object(
            object_name, object_position,
            object_quat=object_quat, scale=scale)
    elif object_name in BULLET_OBJECT_SPECS.keys():
        obj_id = load_bullet_object(
            object_name, basePosition=object_position,
            baseOrientation=object_quat, globalScaling=scale)
    else:
        print(object_name)
        raise NotImplementedError

    # Potentially change the RGBA color
    # rgba may be None or a len4 tuple of floats between 0, 1
    # (1, 1, 1, 1) means we keep the current color
    # (0, 0, 1 ,1) means we make the object very dark blue.
    if rgba is not None:
        # load white texture so that the color shows properly
        blank_texture_id = p.loadTexture(
            os.path.join(
                SHAPENET_ASSET_PATH,
                "ShapeNetCore.v2/03593526/6a13375f8fce3142e6597d391ab6fcc1/images/texture0.png")
        )
        p.changeVisualShape(
            obj_id, -1, textureUniqueId=blank_texture_id, rgbaColor=rgba)
    return obj_id


def load_shapenet_object(object_name, object_position,
                         object_quat=(1, -1, 0, 0),  scale=1.0):
    object_path = shapenet_obj_path_map[object_name]
    path = object_path.split('/')
    dir_name = path[-2]
    object_name = path[-1]
    filepath_collision = os.path.join(
        SHAPENET_ASSET_PATH,
        'ShapeNetCore_vhacd/{0}/{1}/model.obj'.format(dir_name, object_name))
    filepath_visual = os.path.join(
        SHAPENET_ASSET_PATH,
        'ShapeNetCore.v2/{0}/{1}/models/model_normalized.obj'.format(
            dir_name, object_name))
    scale = SHAPENET_SCALE * scale * shapenet_path_scaling_map[object_path]
    collisionid = p.createCollisionShape(p.GEOM_MESH,
                                         fileName=filepath_collision,
                                         meshScale=scale * np.array([1, 1, 1]))
    visualid = p.createVisualShape(p.GEOM_MESH, fileName=filepath_visual,
                                   meshScale=scale * np.array([1, 1, 1]))
    body = p.createMultiBody(0.05, collisionid, visualid)
    p.resetBasePositionAndOrientation(body, object_position, object_quat)
    return body


def load_bullet_object(object_name, **kwargs):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    object_specs = BULLET_OBJECT_SPECS[object_name]
    object_specs.update(**kwargs)
    object_id = p.loadURDF(**object_specs)
    return object_id


class ObjectAliaser:
    """
    Allows converting between object names and their specified aliases

    Assumes that aliases are distinct, but object names need not be.
    Assumes that object names are substrings of their respective aliases.
    """
    def __init__(self, obj_names):
        self._flush_mappings()
        self.obj_names = list(obj_names)
        self._create_aliases_from_obj_names(self.obj_names)

    def _flush_mappings(self):
        self.alias_to_obj_name_map = dict()
        self.obj_names = []

    def _get_obj_name_from_alias(self, alias):
        return self.alias_to_obj_name_map[alias]

    def _create_aliases_from_obj_names(self, obj_names):
        """
        Ex: obj_names = ["conic_cup", "conic_cup", "fountain_vase"]
        get_aliases_from_obj_names(obj_names)
        ==> ["conic_cup_0", "conic_cup_1", fountain_vase]
        """
        self.aliases = []
        obj_name_counter = Counter(obj_names)
        num_of_obj_name_seen_so_far_counter = Counter()
        for obj_name in obj_names:
            if obj_name_counter[obj_name] == 1:
                # No need to alias
                alias = obj_name
            elif obj_name_counter[obj_name] > 1:
                idx = num_of_obj_name_seen_so_far_counter[obj_name]
                alias = f"{obj_name}_{idx}"
            num_of_obj_name_seen_so_far_counter[obj_name] += 1
            self.alias_to_obj_name_map[alias] = obj_name
            self.aliases.append(alias)
        return self.aliases

    def get_first_alias_of_obj_name(self, obj_name):
        for alias in self.aliases:
            if obj_name in alias:
                return alias
        raise ValueError

    def get_aliases(self):
        return self.aliases

    def get_obj_names(self):
        return self.obj_names


BULLET_OBJECT_SPECS = dict(
    duck=dict(
        fileName='duck_vhacd.urdf',
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
    ),
    bowl_small=dict(
        fileName=os.path.join(BASE_ASSET_PATH, 'bowl/bowl.urdf'),
        basePosition=(.72, 0.23, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.07,
    ),
    drawer=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'drawer/drawer_with_tray_inside.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.1,
    ),
    drawer_no_handle=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'drawer/drawer_no_handle.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.1,
    ),
    tray=dict(
        fileName='tray/tray.urdf',
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    low_tray=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'tray/low_tray.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    low_tray_big=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH,'tray/low_tray_big.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    low_tray_big_half_green=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH,
            'tray/low_tray_big_half/low_tray_big_half_green.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    low_tray_big_half_red=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH,
            'tray/low_tray_big_half/low_tray_big_half_red.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    open_box=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'box_open_top/box_open_top.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    long_open_box=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH,
            'box_open_top/long_box_open_top_v3_low_friction.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    cube=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'cube/cube.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.05,
    ),
    spam=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'spam/spam.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.25,
    ),
    pan_tefal=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'dinnerware/pan_tefal.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    table_top=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'table/table2.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    checkerboard_table=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'table_square/table_square.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    torus=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'torus/torus.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    cube_concave=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'cube_concave.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    plate=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'dinnerware/plate.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    husky=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'husky/husky.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    marble_cube=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'marble_cube.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    basket=dict(
        fileName=os.path.join(
            BULLET3_ASSET_PATH, 'dinnerware/cup/cup_small.urdf'),
        basePosition=(.65, 0.3, -.3),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.8,
        useFixedBase=1,
    ),
    button=dict(
        fileName=os.path.join(
            BASE_ASSET_PATH, 'button/button.urdf'),
        basePosition=(.7, 0.2, -.35),
        baseOrientation=(0, 0, 0.707107, 0.707107),
        globalScaling=0.1,
    ),
)
