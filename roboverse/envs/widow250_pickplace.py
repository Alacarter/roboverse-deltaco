import numpy as np
import gym
import random

from .multi_object import (
    MultiObjectConcurrentMultiContainerV2Env,
    MultiObjectConcurrentMultiContainerExtraIdentifiersV2Env,
    MultiIdenticalObjectConcurrentMultiContainerExtraIdentifiersV2Env,
    MultiObjectConcurrentMultiContainerExtraIdentifiersV3Env,
    MultiObjectConcurrentMultiContainerExtraIdentifiersAmbigV3Env,
)
from roboverse.bullet import object_utils, control
import roboverse.bullet as bullet
from roboverse.envs.widow250 import Widow250Env
from roboverse.envs import objects
from roboverse.assets.shapenet_object_lists import CONTAINER_CONFIGS


class Widow250PickPlaceEnv(Widow250Env):

    def __init__(self,
                 container_name='bowl_small',
                 fixed_container_position=False,
                 container_position_z_offset=0.01,
                 **kwargs
                 ):
        self.fixed_container_position = fixed_container_position
        self._load_container_params(
            container_name, container_position_z_offset)
        super(Widow250PickPlaceEnv, self).__init__(**kwargs)
        if isinstance(container_name, str):
            self.num_containers = 1

    def _load_container_params(
            self, container_name, container_position_z_offset):
        self.container_name = container_name

        container_config = CONTAINER_CONFIGS[self.container_name]
        if self.fixed_container_position:
            self.container_position_low = (
                container_config['container_position_default'])
            self.container_position_high = (
                container_config['container_position_default'])
        else:
            self.container_position_low = (
                container_config['container_position_low'])
            self.container_position_high = (
                container_config['container_position_high'])
        self.container_position_z = container_config['container_position_z']
        self.container_orientation = container_config['container_orientation']
        self.container_scale = container_config['container_scale']
        self.min_distance_from_object = (
            container_config['min_distance_from_object'])

        self.place_success_height_threshold = (
            container_config['place_success_height_threshold'])
        self.place_success_radius_threshold = (
            container_config['place_success_radius_threshold'])
        if 'half_extents' in container_config:
            self.container_half_extents = container_config['half_extents']
        else:
            self.container_half_extents = None

        self.container_position_z_offset = container_position_z_offset

    def _load_meshes(self):
        self.table_id = objects.table()
        if self.load_tray:
            self.tray_id = objects.tray_no_divider()
        self.robot_id = objects.widow250()
        self.objects = {}

        assert self.container_position_low[2] == self.object_position_low[2]

        if self.num_objects == 2:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_v2(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_small_obj=0.07,
                    min_distance_large_obj=self.min_distance_from_object,
                )
        elif self.num_objects == 1:
            self.container_position, self.original_object_positions = \
                object_utils.generate_object_positions_single(
                    self.object_position_low, self.object_position_high,
                    self.container_position_low, self.container_position_high,
                    min_distance_large_obj=self.min_distance_from_object,
                )
        else:
            raise NotImplementedError

        self.container_position[-1] = (
            self.container_position_z + self.container_position_z_offset)
        self.container_id = object_utils.load_object(self.container_name,
                                                     self.container_position,
                                                     self.container_orientation,
                                                     self.container_scale)
        bullet.step_simulation(self.num_sim_steps_reset)
        for obj_alias, obj_name, obj_position, obj_rgba in zip(self.object_aliases,
                                                               self.object_names,
                                                               self.original_object_positions,
                                                               self.object_rgbas):
            self.objects[obj_alias] = object_utils.load_object(
                obj_name,
                obj_position,
                object_quat=self.object_orientations[obj_alias],
                scale=self.object_scales[obj_alias],
                randomize_object_quat=self.randomize_object_quat,
                rgba=obj_rgba)
            bullet.step_simulation(self.num_sim_steps_reset)

    def reset(self):
        super(Widow250PickPlaceEnv, self).reset()
        ee_pos_init, ee_quat_init = bullet.get_link_state(
            self.robot_id, self.end_effector_index)
        ee_pos_init[2] -= 0.05
        return self.get_observation()

    def get_reward(self, info):
        if self.reward_type == 'pick_place':
            reward = float(info['place_success_target'])
        elif self.reward_type == 'grasp':
            reward = float(info['grasp_success_target'])
        else:
            raise NotImplementedError
        return reward

    def get_info(self):
        info = super(Widow250PickPlaceEnv, self).get_info()
        info = self.add_place_success_info(info)
        return info

    def add_place_success_info(self, info):
        info['place_success'] = False
        info['place_success_object_name'] = None
        for object_alias in self.object_aliases:
            place_success = object_utils.check_in_container(
                object_alias, self.objects, self.container_position,
                self.place_success_height_threshold,
                self.place_success_radius_threshold)
            if place_success:
                info['place_success'] = place_success
                info['place_success_object_name'] = object_alias

        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, self.container_position,
            self.place_success_height_threshold,
            self.place_success_radius_threshold)

        return info

    def get_obj_positions(self):
        object_infos = {}
        for object_alias in self.object_aliases:
            object_pos, _ = control.get_object_position(
                self.objects[object_alias])
            object_infos[object_alias] = object_pos
        return object_infos


class Widow250PickPlaceConcurrentMultiContainerEnv(Widow250PickPlaceEnv):
    """
    Widow250PickPlaceEnv, but with multiple containers on the
    scene simultaneously
    """
    def __init__(self,
                 container_names=['low_tray_big_half_green'],
                 fixed_container_position=True,
                 random_quadrant_cont_obj_positions=False,
                 container_position_z_offset=0.01,
                 min_distance_between_objs=0.07,
                 **kwargs):
        assert fixed_container_position
        self.fixed_container_position = fixed_container_position

        self.random_quadrant_cont_obj_positions = random_quadrant_cont_obj_positions
        # Place containers in random quadrants and objects in untaken quadrants
        if self.random_quadrant_cont_obj_positions:
            self._set_quadrant_maps()
            self.idf_to_quadrants_map = {}

        self.container_names = container_names
        self.min_distance_between_objs = min_distance_between_objs
        super().__init__(
            object_position_low=[0.63, .14, -0.3],
            object_position_high=[0.73, 0.35, -0.3],
            camera_distance=0.37,
            container_name=container_names,
            fixed_container_position=fixed_container_position,
            container_position_z_offset=container_position_z_offset,
            **kwargs)
        self.tray_position = (np.array(self.object_position_low) +
                              np.array(self.object_position_high)) / 2
        self.tray_half_extents = (
            np.array(self.object_position_high) - self.tray_position)[:2]

    def _load_container_params(
            self, container_names, container_position_z_offset):
        self.container_name_to_params_map = {}
        self.container_name_to_half_extents_map = {}
        for container_name in container_names:
            container_params = dict(CONTAINER_CONFIGS[container_name])
            if self.fixed_container_position:
                container_params['container_position_low'] = list(
                    container_params['container_position_default'])
                container_params['container_position_high'] = list(
                    container_params['container_position_default'])

            for key_suffix in ['low', 'high', 'default']:
                container_params[f'container_position_{key_suffix}'] = list(
                    container_params[f'container_position_{key_suffix}'])
            container_params['container_position_z_offset'] = (
                container_position_z_offset)

            self.container_name_to_half_extents_map[container_name] = (
                container_params['half_extents'])

            self.container_name_to_params_map[container_name] = container_params

    def _set_quadrant_maps(self):
        """
        Defines:
          self.quadrants
          self.quadrant_to_container_pos_map
          self.quadrant_to_obj_pos_range
        """
        assert len(self.container_names) == 2

        self.quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
        origin = np.array([.515, 0.18, -.30])
        dx = np.array([0.16, 0, 0])
        dy = np.array([0, 0.14, 0])

        self.quadrant_to_container_pos_map = {
            (0, 0): origin,
            (0, 1): origin + dy,
            (1, 0): origin + dx,
            (1, 1): origin + dx + dy,
        }

        origin = np.array([0.44, .14, -0.3])
        dx = np.array([0.15, 0, 0])
        dy = np.array([0, 0.12, 0])
        pad = 0.038

        self.quadrant_to_obj_pos_range = {}
        for quadrant in self.quadrants:
            x_coef, y_coef = quadrant
            low = origin + (x_coef * dx) + (y_coef * dy)
            high = low + dx + dy
            low_pad = np.array([x_coef, y_coef, 0]) * pad
            high_pad = np.array([1 - x_coef, 1 - y_coef, 0]) * -pad
            low = low + low_pad
            high = high + high_pad
            self.quadrant_to_obj_pos_range[quadrant] = dict(
                low=list(low),
                high=list(high),
            )

    def _reset_container_and_obj_location_ranges(self):
        """
        Set self.container_name_to_position_map.
        In the `self.random_quadrant_cont_obj_positions` case, also set
        `self.object_position_low` and `self.object_position_high`.
        """
        self.container_name_to_position_map = {}
        if self.random_quadrant_cont_obj_positions:
            cont_quadrant_indices = np.random.choice(
                range(len(self.quadrants)), size=(len(self.container_names),),
                replace=False)
            self.cont_quadrants = np.array(self.quadrants)[cont_quadrant_indices]
            self.cont_quadrants = [
                tuple(cont_quadrant) for cont_quadrant in self.cont_quadrants]
            for container_name, cont_quadrant in zip(
                    self.container_names, self.cont_quadrants):
                container_position = self.quadrant_to_container_pos_map[cont_quadrant]
                container_params = self.container_name_to_params_map[container_name]
                container_position[-1] = (
                    container_params['container_position_z'] +
                    container_params['container_position_z_offset'])
                self.container_name_to_position_map[container_name] = tuple(
                    container_position)
            object_quadrants = list(self.quadrants)
            for cont_quadrant in self.cont_quadrants:
                object_quadrants.remove(cont_quadrant)
            self.object_position_low = [
                self.quadrant_to_obj_pos_range[quadrant]["low"]
                for quadrant in object_quadrants]
            self.object_position_high = [
                self.quadrant_to_obj_pos_range[quadrant]["high"]
                for quadrant in object_quadrants]
        else:
            for container_name in self.container_names:
                container_params = self.container_name_to_params_map[container_name]
                assert (container_params['container_position_low'][2] ==
                        self.object_position_low[2])
                container_position = container_params['container_position_default']
                container_position[-1] = (
                    container_params['container_position_z'] +
                    container_params['container_position_z_offset'])
                self.container_name_to_position_map[container_name] = container_position

        self.object_to_range_idx_list = [] # used by child classes.

    def _load_meshes(self):
        self.table_id = objects.table()
        if self.load_tray:
            self.tray_id = objects.tray_no_divider()
        self.robot_id = objects.widow250()
        self.objects = {}

        self._reset_container_and_obj_location_ranges()

        self.original_object_positions = object_utils.generate_object_positions(
            self.object_position_low, self.object_position_high,
            self.num_objects, min_distance=self.min_distance_between_objs,
            object_to_range_idx_list=self.object_to_range_idx_list)

        self.container_name_to_id_map = {}
        for container_name in self.container_names:
            container_params = self.container_name_to_params_map[container_name]
            container_position = self.container_name_to_position_map[container_name]
            self.container_name_to_id_map[container_name] = \
                object_utils.load_object(
                    container_name,
                    container_position,
                    container_params['container_orientation'],
                    container_params['container_scale'])

        bullet.step_simulation(self.num_sim_steps_reset)

        for obj_alias, obj_name, obj_position, obj_rgba in zip(self.object_aliases,
                                                               self.object_names,
                                                               self.original_object_positions,
                                                               self.object_rgbas):
            self.objects[obj_alias] = object_utils.load_object(
                obj_name,
                obj_position,
                object_quat=self.object_orientations[obj_alias],
                scale=self.object_scales[obj_alias],
                randomize_object_quat=self.randomize_object_quat,
                rgba=obj_rgba)
            bullet.step_simulation(self.num_sim_steps_reset)

    def get_info(self):
        info = super(Widow250PickPlaceConcurrentMultiContainerEnv, self).get_info()
        return info

    def add_place_success_info(self, info):
        def is_place_success(object_name, container_name):
            container_position = self.container_name_to_position_map[container_name]
            half_extents = self.container_name_to_half_extents_map[container_name]
            place_success_height_threshold = (
                self.container_name_to_params_map[container_name]['place_success_height_threshold'])
            place_success_radius_threshold = (
                self.container_name_to_params_map[container_name]['place_success_radius_threshold'])
            place_success = object_utils.check_in_container(
                object_name, self.objects, container_position,
                place_success_height_threshold,
                place_success_radius_threshold,
                container_half_extents=half_extents)
            return place_success

        # place_success = True iff any object is in any container
        # --place_success_object_name = last obj seen that was in any container
        # place_success_target_container = True iff any obj went into target container
        # --place_success_target_container_object_name = last obj iterated
        #     through that was in the target container
        # place_success_target_obj = True iff target obj went into any container
        # --place_success_target_obj_container_name = container that target obj is in
        # place_success_target_obj_target_container = True iff Target obj went
        #   into target container

        info['place_success'] = False
        info['place_success_object_alias'] = None
        info['place_success_target_container'] = False
        info['place_success_target_container_object_alias'] = None
        info['place_success_target_obj'] = False
        info['place_success_target_obj_container_name'] = None
        info['place_success_target_obj_target_container'] = False
        info['num_objs_placed'] = 0
        for container_name in self.container_names:
            is_target_container = container_name == self.target_container
            for object_alias in self.object_aliases:
                is_target_obj = object_alias == self.target_object
                place_success = is_place_success(object_alias, container_name)
                info['num_objs_placed'] += int(place_success)
                if place_success:
                    info['place_success'] = True
                    info['place_success_object_alias'] = object_alias
                    if is_target_obj and is_target_container:
                        info['place_success_target_obj_target_container'] = True
                    if is_target_obj:
                        info['place_success_target_obj'] = True
                        info['place_success_target_obj_container_name'] = container_name
                    if is_target_container:
                        info['place_success_target_container'] = True
                        info['place_success_target_container_object_alias'] = object_alias
        return info

    def get_reward(self, info):
        if self.reward_type == 'pick_place':
            reward = float(info['place_success_target_obj_target_container'])
        else:
            raise NotImplementedError
        return reward


class Widow250PickPlaceMultiObjectConcurrentMultiContainerV2Env(
        MultiObjectConcurrentMultiContainerV2Env,
        Widow250PickPlaceConcurrentMultiContainerEnv):
    """
    Widow250PickPlaceConcurrentMultiContainerEnv but with multiple objects on scene
    """


class Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersV2Env(
        MultiObjectConcurrentMultiContainerExtraIdentifiersV2Env,
        Widow250PickPlaceConcurrentMultiContainerEnv):
    """
    2022.07.29
    Multiple containers simultaneously on the scene, but
    allow each container to be identified with additional identifiers (front/back/left/right)
    """
    def __init__(self, *args, **kwargs):
        self.cont_idf_pairs = [["front", "back"], ["left", "right"]]
        return super().__init__(*args, **kwargs)

    def _make_idf_to_quadrants_map(self):
        assert len(self.idf_to_quadrants_map) == 0

        idfs = ["front", "back", "left", "right"]
        for idf in idfs:
            self.idf_to_quadrants_map[idf] = []

        for quadrant in self.quadrants:
            x, y = quadrant
            if x == 0:
                self.idf_to_quadrants_map['right'].append(quadrant)
            elif x == 1:
                self.idf_to_quadrants_map['left'].append(quadrant)
            if y == 0:
                self.idf_to_quadrants_map['back'].append(quadrant)
            elif y == 1:
                self.idf_to_quadrants_map['front'].append(quadrant)

    def get_target_obj_directional_descriptor(self, task_idx):
        # Used by child classes for get_cont_and_obj_quadrants()
        return None

    def get_cont_and_obj_quadrants(self, target_container_idf=None):
        """Set container and object quadrants."""
        target_obj_directional_descriptor = (
            self.get_target_obj_directional_descriptor(self.task_idx))
        assert target_container_idf in [None] + self.extra_container_idfs

        def create_idf_to_quadrant_map():
            idf_to_quadrant_map = {}
            for idf in self.extra_container_idfs:
                cont_quadrant_idx = np.random.choice(
                    range(len(self.idf_to_quadrants_map[idf])))
                cont_quadrant = self.idf_to_quadrants_map[idf][cont_quadrant_idx]
                idf_to_quadrant_map[idf] = cont_quadrant
            return idf_to_quadrant_map

        container_idf_to_cont_quadrants_map = {}

        if (target_container_idf is None and
                target_obj_directional_descriptor is None):
            # randomly choose container quadrants
            cont_quadrant_indices = np.random.choice(
                range(len(self.quadrants)), size=(len(self.container_names),),
                replace=False)
            cont_quadrants = np.array(self.quadrants)[cont_quadrant_indices]
            cont_quadrants = [
                tuple(cont_quadrant) for cont_quadrant in cont_quadrants]
            obj_quadrants = [
                q for q in self.quadrants if q not in cont_quadrants]
        elif target_obj_directional_descriptor != None:
            # set obj_quadrants first before container quadrants
            idf_to_quadrant_map = create_idf_to_quadrant_map()

            # Ex: target_obj_directional_descriptor = "front"
            obj_directional_pair = [
                idf_pair for idf_pair in self.cont_idf_pairs
                if target_obj_directional_descriptor in idf_pair][0]
            # Ex: obj_directional_pair = ["front", "back"]
            non_target_obj_directional = list(
                set(obj_directional_pair) -
                set([target_obj_directional_descriptor]))[0]
            # Ex: non_target_obj_directional = "back"
            obj_directional_to_obj_quadrants_map = dict([
                (idf, quadrant)
                for idf, quadrant in idf_to_quadrant_map.items()
                if idf in obj_directional_pair])

            # Make sure target obj quadrant comes first (index 0).
            obj_quadrants = [
                obj_directional_to_obj_quadrants_map[target_obj_directional_descriptor],
                obj_directional_to_obj_quadrants_map[non_target_obj_directional]]

            cont_quadrants = [
                q for q in self.quadrants if q not in obj_quadrants]
            random.shuffle(cont_quadrants)

            # Note that this supports cases when either
            # target_obj_directional_descriptor == target_container_idf
            # or target_obj_directional_descriptor != target_container_idf
            if target_container_idf is not None:
                # Create container_idf_to_cont_quadrants_map
                container_idf_pair = [
                    idf_pair for idf_pair in self.cont_idf_pairs
                    if target_container_idf in idf_pair][0]
                for quad in cont_quadrants:
                    for idf in container_idf_pair:
                        if quad in self.idf_to_quadrants_map[idf]:
                            container_idf_to_cont_quadrants_map[idf] = quad
        elif target_container_idf is not None:
            # choose container quadrants based on the position constraint.
            idf_to_quadrant_map = create_idf_to_quadrant_map()
            container_idf_pair = [
                idf_pair for idf_pair in self.cont_idf_pairs
                if target_container_idf in idf_pair][0]
            container_idf_to_cont_quadrants_map = dict([
                (idf, quadrant) for idf, quadrant in idf_to_quadrant_map.items()
                if idf in container_idf_pair])
            cont_quadrants = list(container_idf_to_cont_quadrants_map.values())
            obj_quadrants = [
                q for q in self.quadrants if q not in cont_quadrants]

        return cont_quadrants, obj_quadrants, container_idf_to_cont_quadrants_map

    def _reset_container_and_obj_location_ranges(self):
        """
        Set self.container_name_to_position_map.
        In the `self.random_quadrant_cont_obj_positions` case, also set
        `self.object_position_low` and `self.object_position_high`.
        """
        def create_container_name_to_position_map(
                container_name_to_quadrants_map):
            container_name_to_position_map = {}
            for container_name, quadrant in container_name_to_quadrants_map.items():
                container_position = self.quadrant_to_container_pos_map[quadrant]
                container_params = self.container_name_to_params_map[container_name]
                container_position[-1] = (
                    container_params['container_position_z']
                    + container_params['container_position_z_offset'])
                container_name_to_position_map[container_name] = tuple(
                    container_position)
            return container_name_to_position_map

        assert self.random_quadrant_cont_obj_positions
        if self.idf_to_quadrants_map == {}:
            self._make_idf_to_quadrants_map()

        self.container_name_to_position_map = {}

        target_container_idf_idx = self.task_idx // self.num_obj_idfs

        # Choose quadrants for containers
        if 0 <= target_container_idf_idx < self.num_containers:
            self.cont_quadrants, self.object_quadrants, _ = (
                self.get_cont_and_obj_quadrants(target_container_idf=None))
            container_name_to_quadrants_map = dict(
                zip(self.container_names, self.cont_quadrants))
            try:
                # No need to define this when calling load_meshes the
                # first time (which is when this errors)
                self.target_cont_quadrant = container_name_to_quadrants_map[self.target_container]
            except:
                pass
            self.container_name_to_position_map = create_container_name_to_position_map(
                container_name_to_quadrants_map)
        elif self.num_containers <= target_container_idf_idx < self.num_container_idfs:
            self.cont_quadrants, self.object_quadrants, container_idf_to_cont_quadrants_map = (
                self.get_cont_and_obj_quadrants(
                    target_container_idf=self.target_container_idf))

            self.target_cont_quadrant = (
                container_idf_to_cont_quadrants_map[self.target_container_idf])
            non_target_cont_quadrant = [
                quadrant
                for idf, quadrant in container_idf_to_cont_quadrants_map.items()
                if quadrant != self.target_cont_quadrant][0]

            # Assign the target container to the target_container_idf quadrant
            # Assign the non-target container to the
            # non-target_container_idf quadrant
            non_target_container = [
                cont_name for cont_name in self.container_names
                if cont_name != self.target_container][0]
            container_name_to_quadrants_map = {}
            container_name_to_quadrants_map[self.target_container] = (
                self.target_cont_quadrant)
            container_name_to_quadrants_map[non_target_container] = (
                non_target_cont_quadrant)
            self.container_name_to_position_map = (
                create_container_name_to_position_map(
                    container_name_to_quadrants_map))
        else:
            raise NotImplementedError

        self.object_position_low = [
            self.quadrant_to_obj_pos_range[quadrant]["low"]
            for quadrant in self.object_quadrants]
        self.object_position_high = [
            self.quadrant_to_obj_pos_range[quadrant]["high"]
            for quadrant in self.object_quadrants]

        self.object_to_range_idx_list = []

    def get_observation(self):
        def create_multihot_from_quadrant_list(quadrants_list):
            dim = len(self.quadrants)
            vector = np.zeros((dim,))
            for quadrant in quadrants_list:
                idx = self.quadrants.index(quadrant)
                vector[idx] = 1.0
            return vector

        observation = super().get_observation()

        if hasattr(self, "target_container"):
            observation["target_container_pos_xy"] = np.array(
                self.container_name_to_position_map[self.target_container])[:2]

        if hasattr(self, "target_object"):
            observation["target_object_pos_xy"] = bullet.get_object_position(
                self.objects[self.target_object])[0][:2]

        observation["container_quadrants_onehot"] = (
            create_multihot_from_quadrant_list(self.cont_quadrants))
        observation["target_container_quadrant_onehot"] = (
            create_multihot_from_quadrant_list([self.target_cont_quadrant]))

        return observation

    def _set_observation_space(self):
        super()._set_observation_space()
        cont_eps = 0.02
        target_cont_pos_xy_space = gym.spaces.Box(
            np.array([.515, .18]) - cont_eps,
            np.array([.515 + .16, .18 + .14]) + cont_eps)

        obj_eps = 0.1 # for error
        target_obj_pos_xy_space = gym.spaces.Box(
            np.array([.44, .14]) - obj_eps,
            np.array([.44 + 2 * .15, .14 + 2 * .12]) + obj_eps)
        spaces = {
            'target_container_pos_xy': target_cont_pos_xy_space,
            'target_object_pos_xy': target_obj_pos_xy_space}
        self.observation_space.spaces.update(spaces)


class Widow250PickPlaceMultiIdenticalObjectConcurrentMultiContainerExtraIdentifiersV2Env(
        MultiIdenticalObjectConcurrentMultiContainerExtraIdentifiersV2Env,
        Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersV2Env):
    """
    2022.11.15
    Widow250PickPlaceMultiObjectV2Env but with visually identical objects
    on the scene that are distinguished only by the language.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _reset_container_and_obj_location_ranges(self):
        """
        Set the quadrant position object locations for each object
        individually.
        """
        def set_object_to_range_idx_list():
            self.direction_to_quadrants = {
                "front": [(0, 1), (1, 1)],
                "back": [(0, 0), (1, 0)],
                "left": [(1, 0), (1, 1)],
                "right": [(0, 0), (0, 1)],
            }
            target_container_idf_idx = self.task_idx // self.num_obj_idfs
            if 0 <= target_container_idf_idx < self.num_containers:
                # target object always in 0th quadrant
                self.object_to_range_idx_list = [0] + [1] * (self.num_objects - 1)
                # indicates elem of self.object_quadrants for each elem of
                # self.object_aliases
            elif self.num_containers <= target_container_idf_idx < self.num_container_idfs:
                # deterministically set it
                target_obj_directional_descriptor = (
                    self.get_target_obj_directional_descriptor(self.task_idx))
                quadrants_satisfying_direction = set(
                    self.direction_to_quadrants[target_obj_directional_descriptor])
                avail_quadrant_for_target_obj = list(
                    set(self.object_quadrants).intersection(
                        quadrants_satisfying_direction))[0]
                target_obj_range_idx = self.object_quadrants.index(
                    avail_quadrant_for_target_obj)
                assert len(self.object_quadrants) == 2
                self.object_to_range_idx_list = (
                    [target_obj_range_idx] +
                    [1 - target_obj_range_idx] * (self.num_objects - 1))
            else:
                raise NotImplementedError

        super()._reset_container_and_obj_location_ranges()
        # Set the quadrants each object needs to be in.
        if self.identical_distractor_mode == "same_color_mesh":
            set_object_to_range_idx_list()
            # the objects need to be in specific quadrants to match language
        elif self.identical_distractor_mode == "same_mesh":
            # Don't set self.object_to_range_idx_list
            # (objects go to random quadrants)
            # since the grounding is with language
            pass


class Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersV3Env(
        MultiObjectConcurrentMultiContainerExtraIdentifiersV3Env,
        Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersV2Env):
    """2022.08.15. Support both extra idfs for objs and containers."""


class Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersAmbigV3Env(
        MultiObjectConcurrentMultiContainerExtraIdentifiersAmbigV3Env,
        Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersV2Env):
    """2022.10.24, 2022.12.18. Support ambiguous language."""

    def _reset_container_and_obj_location_ranges(self):
        super()._reset_container_and_obj_location_ranges()
        # Set the quadrants each object needs to be in.
        # set_object_to_range_idx_list()
        self.object_to_range_idx_list = (
            [0] + [1] * (self.num_ambig_distractors) +
            list(np.random.choice(
                [0, 1], self.num_objects - (1 + self.num_ambig_distractors)))
        )
        # print("self.object_to_range_idx_list", self.object_to_range_idx_list)
        # the objects need to be in specific quadrants to match the language

    def get_cont_and_obj_quadrants(
            self, target_container_idf=None,
            target_obj_directional_descriptor=None):
        cont_quadrants, obj_quadrants, _ = super().get_cont_and_obj_quadrants(
            target_container_idf=target_container_idf)

        # Reorder the obj_quadrants list produced by parent class so that
        # 0th index is the quadrant for the target object
        target_obj_directional_descriptor = (
            self.get_target_obj_directional_descriptor(self.task_idx))
        quadrants_satisfying_direction = set(
            self.idf_to_quadrants_map[target_obj_directional_descriptor])
        avail_quadrant_for_target_obj = list(
            set(obj_quadrants).intersection(quadrants_satisfying_direction))[0]
        target_obj_range_idx = obj_quadrants.index(
            avail_quadrant_for_target_obj)

        # Reorder obj_quadrants so that 0th index is the
        # quadrant for the target object
        obj_quadrants = (
            [obj_quadrants[target_obj_range_idx]] +
            obj_quadrants[:target_obj_range_idx] +
            obj_quadrants[target_obj_range_idx + 1:])

        return cont_quadrants, obj_quadrants, _


if __name__ == "__main__":

    # Fixed container position
    env = Widow250PickPlaceEnv(
        reward_type='pick_place',
        control_mode='discrete_gripper',
        object_names=('shed', 'two_handled_vase'),
        object_scales=(0.7, 0.6),
        target_object='shed',
        load_tray=False,
        object_position_low=(.49, .18, -.20),
        object_position_high=(.59, .27, -.20),

        container_name='cube',
        container_position_low=(.72, 0.23, -.20),
        container_position_high=(.72, 0.23, -.20),
        container_position_z=-0.34,
        container_orientation=(0, 0, 0.707107, 0.707107),
        container_scale=0.05,

        camera_distance=0.29,
        camera_target_pos=(0.6, 0.2, -0.28),
        gui=True
    )

    import time
    for _ in range(10):
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample()*0.1)
            time.sleep(0.1)
