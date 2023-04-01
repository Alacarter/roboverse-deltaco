import argparse
import gym
import numpy as np
import os
from PIL import Image

import roboverse
from roboverse.assets.meta_env_object_lists import (
    PICK_PLACE_TRAIN_TASK_OBJECTS,
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP,
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP,
)
from roboverse.assets.shapenet_object_lists import (
    OBJECT_SCALINGS,
    CONTAINER_CONFIGS,
)
import roboverse.bullet as bullet
from roboverse.bullet.serializable import Serializable
from roboverse.bullet import object_utils
from roboverse.envs import objects


def flatten_list(lst):
    flattened_lst = []
    for elem in lst:
        if isinstance(flattened_lst, list):
            flattened_lst.extend(elem)
        else:
            flattened_lst.append(elem)
    return flattened_lst


objects_to_visualize = PICK_PLACE_TRAIN_TASK_OBJECTS
# objects_to_visualize = flatten_list(objects_to_visualize)[:8]
objects_to_visualize = flatten_list(objects_to_visualize)[:]
NUM_ROWS, NUM_COLS = 4, 8

print("objects_to_visualize", objects_to_visualize)


def get_scaling(object_name):
    obj_scale = OBJECT_SCALINGS.get(object_name, None)
    container_info = CONTAINER_CONFIGS.get(object_name, None)
    if obj_scale is not None:
        return obj_scale
    elif container_info is not None:
        return 0.4 * container_info['container_scale']
    else:
        return 0.75


class ObjectsEnv(gym.Env, Serializable):

    def __init__(self,
                 control_mode='continuous',
                 observation_mode='pixels',
                 observation_img_h=128 * (NUM_ROWS + 2),
                 observation_img_w=128 * (NUM_COLS + 2),
                 transpose_image=True,

                 object_names=tuple(objects_to_visualize),
                 object_scales=tuple([
                    get_scaling(object_name) for object_name
                    in objects_to_visualize]),
                 object_orientations=tuple(
                    [(0, 0, 1, 0)] * len(objects_to_visualize)),
                 object_position_high=(.7, .27, -.30),
                 object_position_low=(.5, .18, -.30),
                 target_object=objects_to_visualize[0],
                 load_tray=True,

                 num_sim_steps=10,
                 num_sim_steps_reset=10,
                 num_sim_steps_discrete_action=75,

                 reward_type='grasping',
                 grasp_success_height_threshold=-0.25,
                 grasp_success_object_gripper_threshold=0.1,

                 xyz_action_scale=0.2,
                 abc_action_scale=20.0,
                 gripper_action_scale=20.0,

                 ee_pos_high=(0.8, .4, -0.1),
                 ee_pos_low=(.4, -.2, -.34),
                 camera_target_pos=(0.75, -0.1, -0.28),
                 camera_distance=0.425,
                 camera_roll=0.0,
                 camera_pitch=-90,
                 camera_yaw=180,

                 gui=False,
                 in_vr_replay=False,

                 obj_name_to_custom_obj_pos_offset_map={},
                 obj_name_to_custom_obj_orientation_map={},
                 obj_name_to_custom_obj_scales={},

                 # For loading objects of same idf close together:
                 idf_to_obj_names_map=None,
                 idf_to_num_obj_per_row=None,
                 idf_to_start_xy=None,
                 ):

        self.control_mode = control_mode
        self.observation_mode = observation_mode
        self.observation_img_h = observation_img_h
        self.observation_img_w = observation_img_w
        self.transpose_image = transpose_image

        self.num_sim_steps = num_sim_steps
        self.num_sim_steps_reset = num_sim_steps_reset
        self.num_sim_steps_discrete_action = num_sim_steps_discrete_action

        self.reward_type = reward_type
        self.grasp_success_height_threshold = grasp_success_height_threshold
        self.grasp_success_object_gripper_threshold = \
            grasp_success_object_gripper_threshold

        self.gui = gui

        self.ee_pos_high = ee_pos_high
        self.ee_pos_low = ee_pos_low

        bullet.connect_headless(self.gui)

        # object stuff
        assert target_object in object_names
        assert len(object_names) == len(object_scales)
        self.load_tray = load_tray
        self.num_objects = len(object_names)
        self.object_position_high = list(object_position_high)
        self.object_position_low = list(object_position_low)
        self.object_names = object_names
        self.target_object = target_object
        self.object_scales = dict()
        self.object_orientations = dict()
        for orientation, object_scale, object_name in \
                zip(object_orientations, object_scales, self.object_names):
            self.object_orientations[object_name] = orientation
            self.object_scales[object_name] = object_scale

        self.obj_name_to_custom_obj_pos_offset_map = obj_name_to_custom_obj_pos_offset_map
        self.obj_name_to_custom_obj_orientation_map = obj_name_to_custom_obj_orientation_map
        self.obj_name_to_custom_obj_scales = obj_name_to_custom_obj_scales

        # For loading objects of same idf close together:
        self.idf_to_obj_names_map = idf_to_obj_names_map
        self.idf_to_num_obj_per_row = idf_to_num_obj_per_row
        self.idf_to_start_xy = idf_to_start_xy

        self.in_vr_replay = in_vr_replay
        # self._load_meshes()

        # self.movable_joints = bullet.get_movable_joints(self.robot_id)
        # self.end_effector_index = END_EFFECTOR_INDEX
        # self.reset_joint_values = RESET_JOINT_VALUES
        # self.reset_joint_indices = RESET_JOINT_INDICES

        self.xyz_action_scale = xyz_action_scale
        self.abc_action_scale = abc_action_scale
        self.gripper_action_scale = gripper_action_scale

        self.camera_target_pos = camera_target_pos
        self.camera_distance = camera_distance
        self.camera_roll = camera_roll
        self.camera_pitch = camera_pitch
        self.camera_yaw = camera_yaw
        view_matrix_args = dict(target_pos=self.camera_target_pos,
                                distance=self.camera_distance,
                                yaw=self.camera_yaw,
                                pitch=self.camera_pitch,
                                roll=self.camera_roll,
                                up_axis_index=2)
        self._view_matrix_obs = bullet.get_view_matrix(
            **view_matrix_args)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.observation_img_h, self.observation_img_w)

        self._set_action_space()
        self._set_observation_space()

        self.is_gripper_open = True

        self.reset()
        # self.ee_pos_init, self.ee_quat_init = bullet.get_link_state(
        #     self.robot_id, self.end_effector_index)

    def _load_objects_in_rectangle(self, object_names, num_obj_per_row, start_xy, dist):
        objects_dict = {}
        x_hi, y_lo = start_xy
        for i, object_name in enumerate(object_names):
            obj_pos_x = x_hi - (dist * (i % num_obj_per_row))
            obj_pos_y = y_lo + (dist * (i // num_obj_per_row))
            obj_pos_z = -0.36
            if object_name in self.obj_name_to_custom_obj_pos_offset_map:
                x_offset, y_offset, z_offset = self.obj_name_to_custom_obj_pos_offset_map[object_name]
                obj_pos_x += x_offset
                obj_pos_y += y_offset
                obj_pos_z += z_offset
            object_position = (obj_pos_x, obj_pos_y, obj_pos_z)

            if object_name in self.obj_name_to_custom_obj_orientation_map:
                obj_quat = self.obj_name_to_custom_obj_orientation_map[object_name]
            else:
                obj_quat = self.object_orientations[object_name]

            obj_scale = self.object_scales[object_name]
            if object_name in self.obj_name_to_custom_obj_scales:
                obj_scale *= self.obj_name_to_custom_obj_scales[object_name]

            objects_dict[object_name] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=obj_quat,
                scale=obj_scale)

        return objects_dict

    def _load_meshes(self):
        self.table_id = objects.table()
        self.objects = self._load_objects_in_rectangle(
            self.object_names, num_obj_per_row=NUM_COLS, start_xy=(1.05, -.25), dist=0.1)
        bullet.step_simulation(self.num_sim_steps_reset)

    def _load_meshes_by_idf(self):
        """Load objects so that those of the same idf are in the same rectangular cluster"""
        self.table_id = objects.table()
        for idf, object_names in self.idf_to_obj_names_map.items():
            num_obj_per_row = self.idf_to_num_obj_per_row[idf]
            start_xy = self.idf_to_start_xy[idf]
            obj_name_to_obj_id = self._load_objects_in_rectangle(object_names, num_obj_per_row, start_xy, dist=0.1)
        bullet.step_simulation(self.num_sim_steps_reset)

    def _set_action_space(self):
        self.action_dim = 0
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (
                self.observation_img_h * self.observation_img_w) * 3
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)
            robot_state_dim = 10  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'image': img_space, 'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def reset(self):
        if self.idf_to_obj_names_map is not None:
            self._load_meshes_by_idf()
        else:
            self._load_meshes()

    def step(self, action):
        return self.render_obs(), None, None, None

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.observation_img_h, self.observation_img_w,
            self._view_matrix_obs, self._projection_matrix_obs, shadow=0)
        if self.transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img


def init_env_and_save_image(env_kwargs):
    env = roboverse.make('PickPlaceTrainObject-v0',
                         gui=True, transpose_image=False, **env_kwargs)

    for i in range(1):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, 0, 0])
        im = Image.fromarray(obs)
        im.save(os.path.join(args.save_path, '{}.png'.format(i)))


def create_train_test_obj_image():
    obj_name_to_custom_obj_pos_offset_map = {
        "conic_cup": (0, -0.02, 0),
        "square_prism_bin": (0, -0.01, 0),
        "pepsi_bottle": (0, 0.005, 0),
        "bullet_vase": (0, 0.015, 0),
        "glass_half_gallon": (0, -0.01, 0),
        "two_layered_lampshade": (0, 0.01, 0),
    }
    obj_name_to_custom_obj_orientation_map = {"modern_canoe": (0.707, 0, 0.707, 0)}
    obj_name_to_custom_obj_scales = {"modern_canoe": 0.5}
    env_kwargs = dict(
        obj_name_to_custom_obj_pos_offset_map=obj_name_to_custom_obj_pos_offset_map,
        obj_name_to_custom_obj_orientation_map=obj_name_to_custom_obj_orientation_map,
        obj_name_to_custom_obj_scales=obj_name_to_custom_obj_scales,
    )
    init_env_and_save_image(env_kwargs)


def create_obj_clustered_by_color():
    # Tune object appearance in the image.
    obj_name_to_custom_obj_pos_offset_map = {
        "conic_cup": (0, -0.01, 0),
        "square_prism_bin": (0, -0.01, 0),
        "pepsi_bottle": (0, 0.005, 0),
        "bullet_vase": (0, 0.015, 0),
        "glass_half_gallon": (0, -0.01, 0),
        "two_layered_lampshade": (0, 0.01, 0),
        "stalagcite_chunk": (0, -0.01, 0),
        "short_handle_cup": (0, 0.005, 0),
    }
    obj_name_to_custom_obj_orientation_map = {
        "modern_canoe": (0.707, 0, 0.707, 0),
        "vintage_canoe": (0.707, 0, 0.707, 0),
    }
    obj_name_to_custom_obj_scales = {"modern_canoe": 0.5}
    idf_to_num_obj_per_row = {
        "white": 3,
        "red": 3,
        "orange": 3,
        "yellow": 3,
        "black and white": 6,
        "blue": 6,
        "brown": 6,
        "gray": 6,
    }
    idf_to_start_xy = {
        "white": (1.15, -.25 + 0.1 * 0),
        "red": (1.15, -.25 + 0.1 * 1),
        "orange": (1.15, -.25 + 0.1 * 2),
        "yellow": (1.15, -.25 + 0.1 * 3),
        "black and white": (0.85, -.25 + 0.1 * 0),
        "blue": (0.65, -.25 + 0.1 * 0),
        "brown": (0.85, -.25 + 0.1 * 1),
        "gray": (0.85, -.25 + 0.1 * 3),
    }
    env_kwargs = dict(
        obj_name_to_custom_obj_pos_offset_map=obj_name_to_custom_obj_pos_offset_map,
        obj_name_to_custom_obj_orientation_map=obj_name_to_custom_obj_orientation_map,
        obj_name_to_custom_obj_scales=obj_name_to_custom_obj_scales,
        idf_to_obj_names_map=PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP,
        idf_to_num_obj_per_row=idf_to_num_obj_per_row,
        idf_to_start_xy=idf_to_start_xy,
    )

    init_env_and_save_image(env_kwargs)


def create_obj_clustered_by_shape():
    # Tune object appearance in the image.
    obj_name_to_custom_obj_pos_offset_map = {
        "conic_cup": (0, -0.01, 0),
        "square_prism_bin": (0, -0.01, 0),
        "pepsi_bottle": (0, 0.005, 0),
        "bullet_vase": (0, 0.015, 0),
        "glass_half_gallon": (0, -0.01, 0),
        "two_layered_lampshade": (0, 0.01, 0),
        "stalagcite_chunk": (0, -0.01, 0),
        "short_handle_cup": (0, 0.005, 0),
    }
    obj_name_to_custom_obj_orientation_map = {
        "modern_canoe": (0.707, 0, 0.707, 0),
        "vintage_canoe": (0.707, 0, 0.707, 0),
    }
    obj_name_to_custom_obj_scales = {"modern_canoe": 0.5}
    idf_to_num_obj_per_row = {
        "vase": 4,
        "freeform": 4,
        "bottle": 4,
        "chalice": 4,
        "canoe": 4,
        "bowl": 5,
        "cup": 5,
        "trapezoidal prism": 5,
        "cylinder": 5,
        "round hole": 5,
    }
    d = 0.1
    idf_to_start_xy = {
        "vase": (1.15, -.25 + 0 * d),
        "freeform": (1.15, -.25 + 2 * d),
        "bottle": (1.15 - 2 * d, -.25 + 2 * d),
        "chalice": (1.15, -.25 + 3 * d),
        "canoe": (1.15 - 2 * d, -.25 + 1 * d),
        "bowl": (1.15 - 4 * d, -.25 + 0 * d),
        "cup": (1.15 - 4 * d, -.25 + 1 * d),
        "trapezoidal prism": (1.15 - 4 * d, -.25 + 2 * d),
        "cylinder": (1.15 - 4 * d, -.25 + 3 * d),
        "round hole": (1.15 - 6 * d, -.25 + 3 * d),
    }
    env_kwargs = dict(
        obj_name_to_custom_obj_pos_offset_map=obj_name_to_custom_obj_pos_offset_map,
        obj_name_to_custom_obj_orientation_map=obj_name_to_custom_obj_orientation_map,
        obj_name_to_custom_obj_scales=obj_name_to_custom_obj_scales,
        idf_to_obj_names_map=PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP,
        idf_to_num_obj_per_row=idf_to_num_obj_per_row,
        idf_to_start_xy=idf_to_start_xy,
    )

    init_env_and_save_image(env_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    # create_train_test_obj_image()
    # create_obj_clustered_by_color()
    create_obj_clustered_by_shape()
    # print([(c, len(l)) for c, l in PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP.items()])