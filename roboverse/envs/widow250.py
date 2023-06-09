import gym
import numpy as np

import roboverse.bullet as bullet
from roboverse.bullet import object_utils
from roboverse.bullet.object_utils import ObjectAliaser
from roboverse.bullet.serializable import Serializable
from roboverse.envs import objects

END_EFFECTOR_INDEX = 8
RESET_JOINT_VALUES = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.036, -0.036]
RESET_JOINT_VALUES_GRIPPER_CLOSED = [1.57, -0.6, -0.6, 0, -1.57, 0., 0., 0.015, -0.015]
RESET_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 7, 10, 11]
GUESS = 3.14  # This is a guess, need to verify what joint this is
JOINT_LIMIT_LOWER = [-3.14, -1.88, -1.60, -3.14, -2.14, -3.14, -GUESS, 0.015,
                     -0.037]
JOINT_LIMIT_UPPER = [3.14, 1.99, 2.14, 3.14, 1.74, 3.14, GUESS, 0.037, -0.015]
JOINT_RANGE = []
for upper, lower in zip(JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER):
    JOINT_RANGE.append(upper - lower)

GRIPPER_LIMITS_LOW = JOINT_LIMIT_LOWER[-2:]
GRIPPER_LIMITS_HIGH = JOINT_LIMIT_UPPER[-2:]
GRIPPER_OPEN_STATE = [0.036, -0.036]
GRIPPER_CLOSED_STATE = [0.015, -0.015]

ACTION_DIM = 8


class Widow250Env(gym.Env, Serializable):

    def __init__(self,
                 control_mode='continuous',
                 observation_mode='pixels',
                 observation_img_dim=48,
                 observation_img_hd_dim=None,
                 transpose_image=True,

                 object_names=('beer_bottle', 'gatorade'),
                 object_scales=(0.75, 0.75),
                 object_orientations=((0, 0, 1, 0), (0, 0, 1, 0)),
                 object_position_high=(.7, .27, -.30),
                 object_position_low=(.5, .18, -.30),
                 target_object='gatorade',
                 load_tray=True,
                 randomize_object_quat=False,

                 num_sim_steps=10,
                 num_sim_steps_reset=50,
                 num_sim_steps_discrete_action=75,

                 reward_type='grasping',
                 grasp_success_height_threshold=-0.25,
                 grasp_success_object_gripper_threshold=0.1,

                 use_neutral_action=False,
                 neutral_gripper_open=True,

                 xyz_action_scale=0.2,
                 abc_action_scale=20.0,
                 gripper_action_scale=2.0,

                 ee_pos_high=(0.8, .4, -0.1),
                 ee_pos_low=(.4, -.2, -.34),
                 camera_target_pos=(0.6, 0.2, -0.28),
                 camera_distance=0.29,
                 camera_roll=0.0,
                 camera_pitch=-40,
                 camera_yaw=180,

                 gui=False,
                 in_vr_replay=False,
                 ):

        self.control_mode = control_mode
        self.observation_mode = observation_mode
        self.observation_img_dim = observation_img_dim
        self.observation_img_hd_dim = observation_img_hd_dim
        self.transpose_image = transpose_image

        self.num_sim_steps = num_sim_steps
        self.num_sim_steps_reset = num_sim_steps_reset
        self.num_sim_steps_discrete_action = num_sim_steps_discrete_action

        self.reward_type = reward_type
        self.grasp_success_height_threshold = grasp_success_height_threshold
        self.grasp_success_object_gripper_threshold = \
            grasp_success_object_gripper_threshold

        self.use_neutral_action = use_neutral_action
        self.neutral_gripper_open = neutral_gripper_open

        self.gui = gui

        self.fc_input_key = 'state'
        self.cnn_input_key = 'image'
        self.terminates = False
        self.scripted_traj_len = 30
        self.gripper_idx = 6

        self.ee_pos_high = ee_pos_high
        self.ee_pos_low = ee_pos_low

        bullet.connect_headless(self.gui)

        # object stuff
        if target_object:
            assert target_object in object_names
        assert len(object_names) == len(object_scales)
        self.load_tray = load_tray
        self.num_objects = len(object_names)
        self.object_position_high = list(object_position_high)
        self.object_position_low = list(object_position_low)
        self.set_object_names(object_names)
        self.target_object = target_object
        self.object_scales = dict()
        self.object_orientations = dict()
        for orientation, object_scale, object_alias in \
                zip(object_orientations, object_scales, self.object_aliases):
            self.object_orientations[object_alias] = orientation
            self.object_scales[object_alias] = object_scale
        self.randomize_object_quat = randomize_object_quat
        self.in_vr_replay = in_vr_replay
        self._load_meshes()

        self.movable_joints = bullet.get_movable_joints(self.robot_id)
        self.end_effector_index = END_EFFECTOR_INDEX
        self.reset_joint_values = RESET_JOINT_VALUES
        self.reset_joint_indices = RESET_JOINT_INDICES

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
            self.observation_img_dim, self.observation_img_dim)

        if self.observation_img_hd_dim is not None:
            self._projection_matrix_obs_hd = bullet.get_projection_matrix(
                self.observation_img_hd_dim, self.observation_img_hd_dim)

        self._set_action_space()
        self._set_observation_space()

        self.is_gripper_open = True

        self.reset()
        self.ee_pos_init, self.ee_quat_init = bullet.get_link_state(
            self.robot_id, self.end_effector_index)

    def set_object_names(self, object_names):
        self.object_names = list(object_names)
        # Alias object names
        self.obj_aliaser = ObjectAliaser(self.object_names)
        self.object_aliases = self.obj_aliaser.get_aliases()
        self.object_rgbas = self.create_object_rgbas()

    def create_object_rgbas(self):
        # Don't change default color.
        object_rgbas = [None] * len(self.object_aliases)
        return object_rgbas

    def _load_meshes(self):
        self.table_id = objects.table()
        self.robot_id = objects.widow250()

        if self.load_tray:
            self.tray_id = objects.tray()

        self.objects = {}
        if self.in_vr_replay:
            object_positions = self.original_object_positions
        else:
            object_positions = object_utils.generate_object_positions(
                self.object_position_low, self.object_position_high,
                self.num_objects,
            )
            self.original_object_positions = object_positions
        for object_alias, object_name, object_position in zip(self.object_aliases,
                                                              self.object_names,
                                                              object_positions):
            self.objects[object_alias] = object_utils.load_object(
                object_name,
                object_position,
                object_quat=self.object_orientations[object_name],
                scale=self.object_scales[object_name],
                randomize_object_quat=self.randomize_object_quat)
            bullet.step_simulation(self.num_sim_steps_reset)

    def reset(self):
        bullet.reset()
        bullet.setup_headless()
        self._load_meshes()
        bullet.reset_robot(
            self.robot_id,
            self.reset_joint_indices,
            self.reset_joint_values)
        self.is_gripper_open = True
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_state = np.asarray([joint_states[-2], joint_states[-1]])
        self.gripper_dist = abs(gripper_state[1] - gripper_state[0])

        return self.get_observation()

    def reset_robot_only(self):
        bullet.step_simulation(self.num_sim_steps_reset)
        bullet.reset_robot(
            self.robot_id,
            self.reset_joint_indices,
            self.reset_joint_values)
        self.is_gripper_open = True

    def step(self, action):

        if np.isnan(np.sum(action)):
            print('action', action)
            raise RuntimeError('Action has NaN entries')

        action = np.clip(action, -1, +1)

        xyz_action = action[:3]  # ee position actions
        abc_action = action[3:self.gripper_idx]  # ee orientation actions
        gripper_action = action[self.gripper_idx]
        neutral_action = action[7]

        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_state = np.asarray([joint_states[-2], joint_states[-1]])
        self.gripper_dist = abs(gripper_state[1] - gripper_state[0])

        target_ee_pos = ee_pos + self.xyz_action_scale * xyz_action
        ee_deg = bullet.quat_to_deg(ee_quat)
        target_ee_deg = ee_deg + self.abc_action_scale * abc_action
        target_ee_quat = bullet.deg_to_quat(target_ee_deg)

        if self.control_mode == 'continuous':
            num_sim_steps = self.num_sim_steps
            target_gripper_state = (gripper_state +
                                    [self.gripper_action_scale * gripper_action,
                                     -self.gripper_action_scale * gripper_action])

        elif self.control_mode == 'discrete_gripper':
            if gripper_action > 0.5 and not self.is_gripper_open:
                num_sim_steps = self.num_sim_steps_discrete_action
                target_gripper_state = GRIPPER_OPEN_STATE
                self.is_gripper_open = True

            elif gripper_action < -0.5 and self.is_gripper_open:
                num_sim_steps = self.num_sim_steps_discrete_action
                target_gripper_state = GRIPPER_CLOSED_STATE
                self.is_gripper_open = False
            else:
                num_sim_steps = self.num_sim_steps
                if self.is_gripper_open:
                    target_gripper_state = GRIPPER_OPEN_STATE
                else:
                    target_gripper_state = GRIPPER_CLOSED_STATE
                # target_gripper_state = gripper_state
        else:
            raise NotImplementedError

        target_ee_pos = np.clip(target_ee_pos, self.ee_pos_low,
                                self.ee_pos_high)
        target_gripper_state = np.clip(
            target_gripper_state, GRIPPER_LIMITS_LOW, GRIPPER_LIMITS_HIGH)

        bullet.apply_action_ik(
            target_ee_pos, target_ee_quat, target_gripper_state,
            self.robot_id,
            self.end_effector_index, self.movable_joints,
            lower_limit=JOINT_LIMIT_LOWER,
            upper_limit=JOINT_LIMIT_UPPER,
            rest_pose=RESET_JOINT_VALUES,
            joint_range=JOINT_RANGE,
            num_sim_steps=num_sim_steps)

        if self.use_neutral_action and neutral_action > 0.5:
            if self.neutral_gripper_open:
                bullet.move_to_neutral(
                    self.robot_id,
                    self.reset_joint_indices,
                    RESET_JOINT_VALUES)
            else:
                bullet.move_to_neutral(
                    self.robot_id,
                    self.reset_joint_indices,
                    RESET_JOINT_VALUES_GRIPPER_CLOSED)

        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        return self.get_observation(), reward, done, info

    def get_observation(self):
        gripper_state = self.get_gripper_state()
        gripper_binary_state = [float(self.is_gripper_open)]
        ee_pos, ee_quat = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        if self.target_object is not None:
            object_position, object_orientation = bullet.get_object_position(
                self.objects[self.target_object])
        if self.observation_mode == 'pixels':
            image_obs = self.render_obs(
                self.observation_img_dim, self._projection_matrix_obs)
            image_obs = np.float32(image_obs.flatten()) / 255.0

            observation = {
                'state': np.concatenate(
                    (ee_pos, ee_quat, gripper_state, gripper_binary_state)),
                'image': image_obs,
            }

            if self.observation_img_hd_dim is not None:
                image_obs_hd = self.render_obs(
                    self.observation_img_hd_dim,
                    self._projection_matrix_obs_hd)
                image_obs_hd = np.float32(image_obs_hd.flatten()) / 255.0

                observation['image_hd'] = image_obs_hd

            if self.target_object is not None:
                observation.update({
                    'object_position': object_position,
                    'object_orientation': object_orientation,
                })
        else:
            raise NotImplementedError

        return observation

    def get_reward(self, info):
        if self.reward_type == 'grasping':
            reward = float(info['grasp_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):

        info = {'grasp_success': False,
                'object_names': self.object_names,
                'object_aliases': self.object_aliases,
                'target_object': self.target_object,
                'reward_type': self.reward_type,
                }

        if hasattr(self, 'task_idx'):
            if self.task_idx is not None:
                info['task_idx'] = self.task_idx

        for object_alias in self.object_aliases:
            grasp_success = object_utils.check_grasp(
                object_alias, self.objects, self.robot_id,
                self.end_effector_index, self.grasp_success_height_threshold,
                self.grasp_success_object_gripper_threshold)
            if grasp_success:
                info['grasp_success'] = True

        if self.target_object is not None:
            info['grasp_success_target'] = object_utils.check_grasp(
                self.target_object, self.objects, self.robot_id,
                self.end_effector_index, self.grasp_success_height_threshold,
                self.grasp_success_object_gripper_threshold)
        return info

    def render_obs(self, obs_img_dim, proj_matrix_obs):
        img, depth, segmentation = bullet.render(
            obs_img_dim, obs_img_dim, self._view_matrix_obs,
            proj_matrix_obs, shadow=0)
        if self.transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def _set_action_space(self):
        self.action_dim = ACTION_DIM
        act_bound = 1
        act_high = np.ones(self.action_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

    def _set_observation_space(self):
        if self.observation_mode == 'pixels':
            self.image_length = (self.observation_img_dim ** 2) * 3
            img_space = gym.spaces.Box(0, 1, (self.image_length,),
                                       dtype=np.float32)
            robot_state_dim = 10  # XYZ + QUAT + GRIPPER_STATE
            obs_bound = 100
            obs_high = np.ones(robot_state_dim) * obs_bound
            state_space = gym.spaces.Box(-obs_high, obs_high)
            spaces = {'image': img_space, 'state': state_space}
            self.observation_space = gym.spaces.Dict(spaces)

            if self.observation_img_hd_dim is not None:
                image_length_hd = (self.observation_img_hd_dim ** 2) * 3
                img_space_hd = gym.spaces.Box(
                    0, 1, (image_length_hd,), dtype=np.float32)
                spaces['image_hd'] = img_space_hd
                self.observation_space = gym.spaces.Dict(spaces)
        else:
            raise NotImplementedError

    def get_gripper_state(self):
        joint_states, _ = bullet.get_joint_states(self.robot_id,
                                                  self.movable_joints)
        gripper_state = np.asarray(joint_states[-2:])
        return gripper_state

    def close(self):
        bullet.disconnect()

    def save_state(self, path):
        bullet.p.saveBullet(path)

    def restore_state(self, path):
        bullet.p.restoreState(fileName=path)


if __name__ == "__main__":
    env = Widow250Env(gui=True)
    import time

    env.reset()

    for i in range(20):
        print(i)
        obs, rew, done, info = env.step(
            np.asarray([-0.05, 0., 0., 0., 0., 0.5, 0.]))
        print("reward", rew, "info", info)
        time.sleep(0.1)

    env.reset()
    time.sleep(1)
    for _ in range(25):
        env.step(np.asarray([0., 0., 0., 0., 0., 0., 0.6]))
        time.sleep(0.1)

    env.reset()
