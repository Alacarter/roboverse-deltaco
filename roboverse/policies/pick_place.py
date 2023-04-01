import random

import numpy as np

from roboverse.assets.shapenet_object_lists import GRASP_OFFSETS
import roboverse.bullet as bullet


class PickPlace:

    def __init__(self, env, pick_height_thresh=-0.31, xyz_action_scale=7.0,
                 pick_point_z=-0.32, drop_point_z=-0.2,
                 pick_point_noise=0.00, drop_point_noise=0.00,
                 drop_container_scheme=None, pick_object_scheme="random"):
        self.env = env
        self.pick_height_thresh = pick_height_thresh
        self.z_std = 0.01
        self.xyz_action_scale = xyz_action_scale
        self.pick_point_z = pick_point_z
        self.drop_point_z = drop_point_z
        self.pick_point_noise = pick_point_noise
        self.drop_point_noise = drop_point_noise

        # For concurrent multiple container envs.
        assert drop_container_scheme in [None, "target", "random"]
        self.drop_container_scheme = drop_container_scheme

        assert pick_object_scheme in ["target", "random"]
        self.pick_object_scheme = pick_object_scheme

        self.reset()

    def reset(self, drop_point=None, object_to_target=None):
        # self.dist_thresh = 0.06 + np.random.normal(scale=0.01)
        if object_to_target is None:
            if self.pick_object_scheme == "random":
                self.object_to_target = self.env.object_aliases[
                    np.random.randint(self.env.num_objects)]
            elif self.pick_object_scheme == "target":
                self.object_to_target = self.env.target_object
        else:
            self.object_to_target = object_to_target

        self.pick_height_thresh_noisy = (
            self.pick_height_thresh + np.random.normal(scale=self.z_std))

        if drop_point is None:
            half_extent_coef = 0.4
            if self.env.num_containers > 1:
                if self.drop_container_scheme == "target":
                    drop_container_name = self.env.target_container
                elif self.drop_container_scheme == "random":
                    drop_container_name = random.choice(
                        self.env.container_names)
                else:
                    raise NotImplementedError

                container_position = (
                    self.env.container_name_to_position_map[drop_container_name])
                half_extents = (
                    self.env.container_name_to_half_extents_map[drop_container_name])
                container_half_extents_z = half_extent_coef * np.concatenate(
                    [half_extents, np.array([0])], axis=0)

                self.drop_point = np.random.uniform(
                    low=container_position - container_half_extents_z,
                    high=container_position + container_half_extents_z)
            elif self.env.container_half_extents is not None:
                container_half_extents_z = half_extent_coef * np.concatenate(
                    [self.env.container_half_extents, np.array([0])], axis=0)
                self.drop_point = np.random.uniform(
                    low=self.env.container_position - container_half_extents_z,
                    high=self.env.container_position + container_half_extents_z)
            else:
                self.drop_point = self.env.container_position
        else:
            self.drop_point = drop_point

        drop_point_z = self.drop_point_z + np.random.normal(scale=self.z_std)
        self.drop_point[2] = drop_point_z

        # Set a lift point which is in the xy midpoint of the pick_point
        # and drop_point
        self.pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]
        self.lift_point = 0.5 * (self.pick_point + self.drop_point)
        self.lift_point[2] = drop_point_z

        self.place_attempted = False

    def set_pickpoint(self):
        pick_point = bullet.get_object_position(
            self.env.objects[self.object_to_target])[0]

        if self.object_to_target in GRASP_OFFSETS.keys():
            pick_point += np.asarray(GRASP_OFFSETS[self.object_to_target])

        pick_point[2] = self.pick_point_z

        return pick_point

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(
            self.env.robot_id, self.env.end_effector_index)
        object_pos, _ = bullet.get_object_position(
            self.env.objects[self.object_to_target])
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        self.pick_point = self.set_pickpoint()
        gripper_pickpoint_dist = np.linalg.norm(self.pick_point - ee_pos)
        gripper_droppoint_dist = np.linalg.norm(self.drop_point - ee_pos)
        done = False
        action_xyz = [0., 0., 0.]
        action_gripper = [0.]

        if self.place_attempted:
            # Avoid pick and place the object again after one attempt
            pass
        elif gripper_pickpoint_dist > 0.02 and self.env.is_gripper_open:
            # move near the object
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            if xy_diff > 0.03:
                action_xyz[2] = 0.0
        elif self.env.is_gripper_open:
            # near the object enough, performs grasping action
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_gripper = [-0.7]
        elif not object_lifted:
            # lifting objects above the height threshold for picking
            action_xyz = (self.lift_point - ee_pos) * self.xyz_action_scale
        elif gripper_droppoint_dist > 0.02:
            # lifted, now need to move towards the container
            action_xyz = (self.drop_point - ee_pos) * self.xyz_action_scale
        else:
            # already moved above the container; drop object
            action_gripper = [0.7]
            self.place_attempted = True

        agent_info = dict(place_attempted=self.place_attempted, done=done)
        action_angles = [0., 0., 0.]
        neutral_action = [0.]
        action = np.concatenate(
            (action_xyz, action_angles, action_gripper, neutral_action))
        return action, agent_info
