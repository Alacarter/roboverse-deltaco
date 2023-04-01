from roboverse.assets.shapenet_object_lists import (
    TRAIN_OBJECTS, OBJECT_SCALINGS, OBJECT_ORIENTATIONS,
    OBJ_RENAME_MAP,
)
from roboverse.assets.meta_env_object_lists import (
    PICK_PLACE_TRAIN_TASK_OBJECTS_TO_COLOR_MAP,
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP,
    PICK_PLACE_TRAIN_TASK_OBJECTS_TO_SHAPE_MAP,
    PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP,
    COLOR_TO_RGBA_MAP, OBJECT_NAME_TO_COLOR_PAIR_MAP,
)
from roboverse.bullet.graph_utils import OrderedVerticesWeightedGraph

import itertools
import numpy as np


class MultiObjectConcurrentMultiContainerV2Env:
    """
    MultiObjectV2Env, but with multiple containers on the env at the same time.
    The task idx dictates the target index.
    """
    def __init__(self,
                 container_names,
                 num_extra_obj_idfs=0,
                 num_tasks=16,
                 num_objects=1,
                 num_objects_range=(1,2),
                 random_distractor_objs=False,
                 possible_objects=TRAIN_OBJECTS[:10],
                 init_task_idx=None,
                 fixed_task=False,
                 obj_cont_matching_scheme='exhaustive',
                 task_str_format="pick_place_obj",
                 all_obj_added_scaling=1.0, # 1.0 = no additional scaling compared to what was specified in shapenet_object_lists.py
                 **kwargs):
        assert obj_cont_matching_scheme in ['exhaustive', 'even', 'odd']
        # exhaustive: n containers, m objects, for a total of nm tasks
        # even/odd: m objects --> m tasks. each object only targeted for one container.
        self.container_names = np.asarray(container_names)
        self.num_extra_obj_idfs = num_extra_obj_idfs
        self.num_tasks = num_tasks
        self.num_objects = num_objects
        self.num_objects_range = num_objects_range
        self.random_distractor_objs = random_distractor_objs
        self.set_num_objects()
        self.num_containers = len(container_names)
        self.set_num_container_idfs()
        self.obj_cont_matching_scheme = obj_cont_matching_scheme
        if self.obj_cont_matching_scheme == "exhaustive":
            num_tasks_from_extra_obj_idfs = (
                self.num_extra_obj_idfs * self.num_container_idfs)
            self.num_possible_objects = int(
                (num_tasks - num_tasks_from_extra_obj_idfs) /
                self.num_container_idfs)
        else:
            assert self.num_containers == 2
            self.num_possible_objects = num_tasks
        self.set_scheme_idx_to_obj_idx_interval_list()

        # Only used in extra obj idf env.
        # This is the number of rows on the task idx chart.
        self.num_obj_idfs = self.num_possible_objects + self.num_extra_obj_idfs

        self.valid_task_str_formats = [
            'pick_place_obj', 'put_place_obj', 'obj_container',
            'obj', 'container_obj', 'container']
        assert task_str_format in self.valid_task_str_formats + ['random']
        self.task_str_format = task_str_format
        self.all_obj_added_scaling = all_obj_added_scaling
        self.possible_objects = np.asarray(possible_objects)
        self.task_idx = init_task_idx
        self.fixed_task = fixed_task
        self.container_idfs_to_language_name_map = {
            'low_tray_big_half_green': 'green tray',
            'low_tray_big_half_red': 'red tray',
        }
        super().__init__(
            container_names=container_names,
            **kwargs)

    def set_scheme_idx_to_obj_idx_interval_list(self):
        pass

    def set_num_objects(self):
        if self.random_distractor_objs or self.num_objects is None:
            self.num_objects = np.random.randint(*self.num_objects_range)
        elif isinstance(self.num_objects, int):
            pass
        else:
            raise NotImplementedError

    def set_num_container_idfs(self):
        if hasattr(self, "num_container_idfs"):
            pass
        else:
            self.num_container_idfs = self.num_containers

    def maybe_update_target_obj_with_alias(self):
        if self.target_object not in self.object_aliases:
            # the target object was aliased
            # get first alias with target_object as substring.
            # presumably this ends in "_0"
            self.target_object = self.obj_aliaser.get_first_alias_of_obj_name(
                self.target_object)

    def reset(self):
        if self.task_idx is None:
            # randomly sample container and objects from given set
            chosen_obj_idx = np.random.randint(0, len(self.possible_objects),
                                               size=self.num_objects)
            self.set_object_names(tuple(self.possible_objects[chosen_obj_idx]))
            self.target_object = self.object_aliases[0]
            self.target_container = self.container_names[0]
            self.target_container_idf = self.target_container
        else:
            assert len(self.possible_objects[0]) == 2
            assert self.task_idx < 2 * self.num_tasks
            self.update_target_obj_and_container(update_scene_objects=True)
        self.maybe_update_target_obj_with_alias()

        self.object_scales = dict()
        self.object_orientations = dict()
        for object_alias, object_name in zip(self.object_aliases, self.object_names):
            self.object_orientations[object_alias] = OBJECT_ORIENTATIONS[object_name]
            self.object_scales[object_alias] = (
                self.all_obj_added_scaling * OBJECT_SCALINGS[object_name])

        res = super().reset()
        return res

    def get_distractor_obj_names(self, target_obj_name, num_distractor_objs):
        if num_distractor_objs == 0:
            return []
        possible_objs_flattened = [
            item for pair in self.possible_objects for item in pair]
        valid_distractor_obj_names = (
            list(set(possible_objs_flattened) - set([target_obj_name])))
        distractor_obj_names = np.random.choice(
            valid_distractor_obj_names, num_distractor_objs, replace=False)
        return list(distractor_obj_names)

    def reset_task(self, task_idx):
        if not self.fixed_task:
            self.task_idx = task_idx
            self.update_target_obj_and_container()

    def get_objs_and_container_idx_from_task_idx(self, task_idx):
        indices_map = dict(
            chosen_obj_idx=int((task_idx % self.num_possible_objects) / 2),
            target_obj_idx=(task_idx % self.num_possible_objects) % 2,
        )

        if self.obj_cont_matching_scheme == "exhaustive":
            indices_map['target_container_idx'] = (
                task_idx // (self.num_tasks // self.num_containers))
        elif self.obj_cont_matching_scheme == "even":
            indices_map['target_container_idx'] = (
                (task_idx % self.num_possible_objects) % self.num_containers)
        elif self.obj_cont_matching_scheme == "odd":
            indices_map['target_container_idx'] = (
                ((task_idx % self.num_possible_objects) + 1) % self.num_containers)
        else:
            raise NotImplementedError
        return indices_map

    def get_target_obj(self, indices_map):
        object_names = tuple(self.possible_objects[indices_map['chosen_obj_idx']])
        target_obj = object_names[indices_map['target_obj_idx']]
        return target_obj

    def update_target_obj_and_container(self, update_scene_objects=False):
        indices_map = self.get_objs_and_container_idx_from_task_idx(self.task_idx)

        if update_scene_objects:
            self.set_object_names(
                tuple(self.possible_objects[indices_map['chosen_obj_idx']]))

        if update_scene_objects and self.random_distractor_objs:
            self.set_num_objects()
            self.target_object = self.get_target_obj(indices_map)
            distractor_obj_names = self.get_distractor_obj_names(
                self.target_object, self.num_objects - 1)
            self.set_object_names(
                [self.target_object] + distractor_obj_names)
        elif not self.random_distractor_objs:
            self.target_object = self.object_names[indices_map['target_obj_idx']]
        self.maybe_update_target_obj_with_alias()

        if "target_container_idf_idx" in indices_map:
            self.target_container_idf = self.container_idfs[
                indices_map["target_container_idf_idx"]]

        # print("indices_map", indices_map)
        if "obj_idf" in indices_map:
            self.target_object_idf = indices_map["obj_idf"]
        elif hasattr(self, "target_object_idf"):
            # If "obj_idf" not in indices_map, delete that attr cuz it's out of date
            delattr(self, "target_object_idf")

        self.target_container = self.container_names[
            indices_map['target_container_idx']]

    def get_target_obj_name_from_str(self, target_object_str):
        if target_object_str in OBJ_RENAME_MAP:
            target_object_name = OBJ_RENAME_MAP[target_object_str]
        else:
            target_object_name = target_object_str.replace("_", " ")
        return target_object_name

    def get_target_obj_and_cont_from_task_idx(self, task_idx):
        indices_map = self.get_objs_and_container_idx_from_task_idx(task_idx)

        object_names = tuple(self.possible_objects[indices_map['chosen_obj_idx']])
        target_object_str = object_names[indices_map['target_obj_idx']]
        target_object_name = self.get_target_obj_name_from_str(target_object_str)
        target_container_str = self.container_idfs[
            indices_map['target_container_idx']]
        target_container_name = self.container_idfs_to_language_name_map[target_container_str]

        return target_object_name, target_container_name

    def get_task_language(self, target_object_name, target_container_name, n=1):
        task_str_format = str(self.task_str_format)

        task_languages = []
        for i in range(n):
            if task_str_format == "random":
                task_str_format = np.random.choice(self.valid_task_str_formats)

            if task_str_format == "pick_place_obj":
                task = f"pick {target_object_name} and place in {target_container_name}"
            elif task_str_format == "put_place_obj":
                task = f"put {target_object_name} in {target_container_name}"
            elif task_str_format == "obj_container":
                task = f"{target_object_name} in {target_container_name}"
            elif task_str_format == "obj":
                task = target_object_name
            elif task_str_format == "container_obj":
                task = f"{target_container_name} with {target_object_name}"
            elif task_str_format == "container":
                task = target_container_name
            else:
                raise NotImplementedError
            task_languages.append(task)

        if len(task_languages) == 1:
            return task_languages[0]
        else:
            return task_languages

    def get_task_lang_dict(self):
        """Each task is a list of strings, not a single string."""
        instructs = []
        target_objs = []
        target_conts = []
        for task_idx in range(self.num_tasks):
            # This call below may cause reselection
            # of target object name for extra obj idfs
            target_object_name, target_container_name = (
                self.get_target_obj_and_cont_from_task_idx(task_idx))
            task = self.get_task_language(target_object_name, target_container_name)
            instructs.append(task)
            target_objs.append(target_object_name)
            target_conts.append(target_container_name)
        # print("instructs", "\n".join([t[0] for t in instructs]))
        task_lang_dict = dict(
            instructs=instructs,
            target_objs=target_objs,
            target_conts=target_conts,
        )
        return task_lang_dict

    def get_observation(self):
        observation = super().get_observation()
        if self.task_idx is not None:
            one_hot_vector = np.zeros((self.num_tasks,))
            one_hot_vector[self.task_idx] = 1.0
            observation['one_hot_task_id'] = one_hot_vector

        return observation

    def save_state(self, path):
        if not self.random_distractor_objs:
            self._saved_task_idx = self.task_idx
            super().save_state(path)

    def restore_state(self, path):
        if not self.random_distractor_objs:
            self.reset_task(self._saved_task_idx)
            super().restore_state(path)


class MultiObjectConcurrentMultiContainerExtraIdentifiersV2Env(
        MultiObjectConcurrentMultiContainerV2Env):
    """
    2022.07.29
    Supports adding extra tasks by referring to the same two containers differently.
    """
    def __init__(self, extra_container_identifiers, **kwargs):
        self.extra_container_idfs = extra_container_identifiers
        self.num_containers = len(kwargs['container_names']) # 2: red and green.
        self.container_idfs = (kwargs['container_names'] +
                               self.extra_container_idfs)
        self.num_container_idfs = len(self.container_idfs)
        super().__init__(**kwargs)
        for extra_cont_idf in self.extra_container_idfs:
            self.container_idfs_to_language_name_map[extra_cont_idf] = f"{extra_cont_idf} tray"
        assert self.obj_cont_matching_scheme == "exhaustive"
        assert self.num_obj_idfs == self.num_tasks // self.num_container_idfs

    def get_target_obj_and_cont_from_task_idx(self, task_idx):
        """Only used for task language str creation"""
        indices_map = self.get_objs_and_container_idx_from_task_idx(task_idx)

        object_names = tuple(self.possible_objects[indices_map['chosen_obj_idx']])
        target_object_str = object_names[indices_map['target_obj_idx']]
        target_object_name = self.get_target_obj_name_from_str(target_object_str)
        target_container_str = self.container_idfs[
            indices_map['target_container_idf_idx']]
        target_container_name = self.container_idfs_to_language_name_map[target_container_str]

        return target_object_name, target_container_name

    def get_objs_and_container_idx_from_task_idx(self, task_idx):
        indices_map = dict(
            chosen_obj_idx = int((task_idx % self.num_obj_idfs)/ 2),
            target_obj_idx = (task_idx % self.num_obj_idfs) % 2,
        )

        # Calculate column idx (on the task idx spreadsheet)
        indices_map['target_container_idf_idx'] = task_idx // self.num_obj_idfs

        if 0 <= indices_map['target_container_idf_idx'] < self.num_containers:
            indices_map['target_container_idx'] = indices_map['target_container_idf_idx']
        elif self.num_containers <= indices_map['target_container_idf_idx'] < self.num_container_idfs:
            indices_map['target_container_idx'] = np.random.choice(
                range(self.num_containers))
        else:
            raise NotImplementedError

        return indices_map


class MultiIdenticalObjectConcurrentMultiContainerExtraIdentifiersV2Env(
        MultiObjectConcurrentMultiContainerExtraIdentifiersV2Env):
    """
    2022.11.15
    Widow250PickPlaceMultiObjectV2Env but with visually identical objects on the scene
    that are distinguished only by the language.
    """
    def __init__(self, identical_distractor_mode="same_color_mesh", **kwargs):
        self.num_identical_objs = 1 # 1 identical distractor
        assert identical_distractor_mode in ["same_color_mesh", "same_mesh"]
        self.identical_distractor_mode = identical_distractor_mode

        if self.identical_distractor_mode == "same_mesh":
            self.color_to_rgba_map = COLOR_TO_RGBA_MAP
            # Making this deterministic so that the task language list is deterministic
            self.object_name_to_color_pair_map = OBJECT_NAME_TO_COLOR_PAIR_MAP

        super().__init__(**kwargs)
        assert set(self.extra_container_idfs) == set(
            ['front', 'back', 'left', 'right'])

    def get_target_obj_and_cont_from_task_idx(self, task_idx):
        """
        Only used for task language str creation
        Make sure that we add a directional adjective to the target object name
        """
        indices_map = self.get_objs_and_container_idx_from_task_idx(task_idx)

        object_names = tuple(self.possible_objects[indices_map['chosen_obj_idx']])
        target_object_str = object_names[indices_map['target_obj_idx']]
        target_object_name = self.get_target_obj_name_from_str(target_object_str)

        if self.identical_distractor_mode == "same_color_mesh":
            target_object_descriptor = self.get_target_obj_directional_descriptor(task_idx)
        elif self.identical_distractor_mode == "same_mesh":
            target_object_descriptor = self.get_target_obj_shape_descriptor(
                target_object_str)
            if target_object_descriptor != "":
                target_object_descriptor += " colored"

        if target_object_descriptor == "":
            target_object_referent = target_object_name
        else:
            target_object_referent = f"{target_object_descriptor} {target_object_name}"

        target_container_str = self.container_idfs[
            indices_map['target_container_idf_idx']]
        target_container_name = self.container_idfs_to_language_name_map[target_container_str]

        return target_object_referent, target_container_name

    def get_target_obj_directional_descriptor(self, task_idx):
        target_container_idf_idx = task_idx // self.num_obj_idfs
        if 0 <= target_container_idf_idx < self.num_containers:
            directional_descriptor = ""
        elif self.num_containers <= target_container_idf_idx < self.num_container_idfs:
            target_container_idf = self.container_idfs[target_container_idf_idx]
            directional_descriptor = target_container_idf
        else:
            raise NotImplementedError
        return directional_descriptor

    def get_target_obj_shape_descriptor(self, target_object_str):
        colors = self.object_name_to_color_pair_map[target_object_str]
        # Take first elem of the key in self.object_name_to_color_pair_map[target_object]
        return colors[0]

    def get_distractor_obj_names(self, target_obj_name, num_distractor_objs):
        """distractor object will include target object (visually identical distractor)"""
        assert num_distractor_objs >= self.num_identical_objs
        if num_distractor_objs == 0:
            return []

        num_visually_distinct_distractor_objs = (
            num_distractor_objs - self.num_identical_objs)
        possible_objs_flattened = [
            item for pair in self.possible_objects for item in pair]
        valid_distractor_obj_names = (
            list(set(possible_objs_flattened) - set([target_obj_name])))
        distractor_obj_names = np.random.choice(
            valid_distractor_obj_names, num_visually_distinct_distractor_objs,
            replace=False)
        return [target_obj_name] + list(distractor_obj_names)

    def create_object_rgbas(self):
        if self.identical_distractor_mode == "same_color_mesh":
            object_rgbas = [None] * self.num_objects
        elif self.identical_distractor_mode == "same_mesh":
            # Create colors list (ex: ['blue', 'green', 'red'])
            # Make sure the first two objects in list have same object name.
            if self.object_names == 3:
                assert self.object_names == (
                    [self.object_names[0], self.object_names[0], self.object_names[2]])
            target_object_name = self.object_names[0]
            try:
                colors = (
                    self.object_name_to_color_pair_map[target_object_name] +
                    [np.random.choice(list(self.color_to_rgba_map.keys()))])
                # object_rgbas = [(0, 0, 0.5, 1)] + [None] * (self.num_objects - 1)
                object_rgbas = [self.color_to_rgba_map[color] for color in colors]
            except:
                object_rgbas = [None] * self.num_objects
        else:
            raise NotImplementedError
        return object_rgbas


class MultiObjectConcurrentMultiContainerExtraIdentifiersV3Env(
        MultiObjectConcurrentMultiContainerExtraIdentifiersV2Env):
    """
    2022.08.14
    Supports adding extra identifiers for objects, such color.
    """
    def __init__(
            self, extra_obj_idf_schemes,
            extra_obj_idf_instr_trailing_tokens=[],
            distractor_obj_hard_mode_prob=0.0,
            deterministic_target_obj_referent=True, **kwargs):
        assert set(extra_obj_idf_schemes).issubset(set(['color', 'shape']))
        assert 0.0 <= distractor_obj_hard_mode_prob <= 1.0
        self.extra_obj_idf_schemes = extra_obj_idf_schemes
        self.extra_obj_idf_instr_trailing_tokens = extra_obj_idf_instr_trailing_tokens
        self.distractor_obj_hard_mode_prob = distractor_obj_hard_mode_prob
        # The below only affects task language list.
        # If deterministic_target_obj_referent=True, then the full object name with all
        # its attributes will always be given as the instruction.
        self.deterministic_target_obj_referent = deterministic_target_obj_referent
        self.set_extra_obj_idf_attrs()
        if self.distractor_obj_hard_mode_prob >= 0.0:
            self.create_hard_mode_distractor_graphs(kwargs['possible_objects'])
        kwargs['num_extra_obj_idfs'] = self.num_extra_obj_idfs
        super().__init__(**kwargs)

    def set_scheme_idx_to_obj_idx_interval_list(self):
        self.scheme_idx_to_obj_idx_interval_list = []
        for i, num_extra_obj_idfs in enumerate(self.num_extra_obj_idfs_list):
            if i == 0:
                start_idx = self.num_possible_objects
            end_idx = start_idx + num_extra_obj_idfs
            interval = (start_idx, end_idx)
            self.scheme_idx_to_obj_idx_interval_list.append(interval)
            start_idx = end_idx

    def set_extra_obj_idf_attrs(self):
        def add_extra_obj_idf_scheme(extra_obj_idf_scheme):
            if extra_obj_idf_scheme == "color":
                obj_name_to_idf_map = PICK_PLACE_TRAIN_TASK_OBJECTS_TO_COLOR_MAP
                idf_to_objs_map = PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP
            elif extra_obj_idf_scheme == "shape":
                obj_name_to_idf_map = PICK_PLACE_TRAIN_TASK_OBJECTS_TO_SHAPE_MAP
                idf_to_objs_map = PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP
            else:
                raise NotImplementedError

            extra_obj_idfs = list(idf_to_objs_map.keys())

            self.obj_name_to_idf_map_list.append(obj_name_to_idf_map)
            self.idf_to_objs_map_list.append(idf_to_objs_map)
            self.extra_obj_idfs_list.append(extra_obj_idfs)
            self.num_extra_obj_idfs_list.append(len(extra_obj_idfs))

        if len(self.extra_obj_idf_schemes) == 0:
            return

        self.obj_name_to_idf_map_list = []
        self.idf_to_objs_map_list = []
        self.extra_obj_idfs_list = []
        self.num_extra_obj_idfs_list = []

        for extra_obj_idf_scheme in self.extra_obj_idf_schemes:
            add_extra_obj_idf_scheme(extra_obj_idf_scheme)

        self.num_extra_obj_idfs = sum(self.num_extra_obj_idfs_list)

    def create_hard_mode_distractor_graphs(self, possible_objects):
        possible_objs_flattened = [
            item for pair in possible_objects for item in pair]

        self.idf_obj_graphs = []
        self.total_obj_graph = None
        for idf_to_objs_map in self.idf_to_objs_map_list:
            idf_obj_graph = OrderedVerticesWeightedGraph(
                possible_objs_flattened,
                clusters_dict=idf_to_objs_map)

            self.idf_obj_graphs.append(idf_obj_graph)

            if self.total_obj_graph is None:
                self.total_obj_graph = idf_obj_graph
            else:
                self.total_obj_graph += idf_obj_graph

        self.total_minus_idf_obj_graphs = []
        for idf_obj_graph in self.idf_obj_graphs:
            total_minus_idf_obj_graph = self.total_obj_graph - idf_obj_graph
            self.total_minus_idf_obj_graphs.append(total_minus_idf_obj_graph)

    def get_distractor_obj_names(self, target_obj_name, num_distractor_objs):
        obj_idx = self.get_obj_idx(self.task_idx)
        coin_flip = np.random.random()
        if coin_flip < self.distractor_obj_hard_mode_prob:
            distractor_obj_fn = self.get_hard_mode_distractor_obj_names
        else:
            distractor_obj_fn = self.get_random_distractor_obj_names
        return distractor_obj_fn(obj_idx, target_obj_name, num_distractor_objs)

    def get_random_distractor_obj_names(
            self, obj_idx, target_obj_name, num_distractor_objs):
        if num_distractor_objs == 0:
            return []
        elif 0 <= obj_idx < self.num_possible_objects:
            return super().get_distractor_obj_names(
                target_obj_name, num_distractor_objs)
        elif self.num_possible_objects <= obj_idx < self.num_obj_idfs:
            target_obj_idf = self.get_obj_idf(self.task_idx)
            extra_obj_idf_scheme_idx = self.get_extra_obj_idf_scheme_idx(
                self.task_idx)
            obj_name_to_idf_map = self.obj_name_to_idf_map_list[extra_obj_idf_scheme_idx]
            valid_distractor_obj_names = [
                obj for obj, idf in obj_name_to_idf_map.items()
                if idf != target_obj_idf]
            distractor_obj_names = np.random.choice(
                valid_distractor_obj_names, num_distractor_objs, replace=False)
            return list(distractor_obj_names)
        else:
            raise NotImplementedError

    def get_hard_mode_distractor_obj_names(
            self, obj_idx, target_obj_name, num_distractor_objs):
        if num_distractor_objs == 0:
            return []
        elif 0 <= obj_idx < self.num_possible_objects:
            return self.total_obj_graph.get_random_maybe_neighbors(
                target_obj_name, num_distractor_objs)
        elif self.num_possible_objects <= obj_idx < self.num_obj_idfs:
            extra_obj_idf_scheme_idx = self.get_extra_obj_idf_scheme_idx(
                self.task_idx)
            graph = self.total_minus_idf_obj_graphs[extra_obj_idf_scheme_idx]
            return graph.get_random_maybe_neighbors(
                target_obj_name, num_distractor_objs)
        else:
            raise NotImplementedError

    def get_target_obj_and_cont_from_task_idx(self, task_idx):
        """Only used for getting the task instruction string."""
        indices_map = self.get_objs_and_container_idx_from_task_idx(task_idx)
        obj_idx = self.get_obj_idx(task_idx)

        object_names = tuple(self.possible_objects[indices_map['chosen_obj_idx']])
        target_object_str = object_names[indices_map['target_obj_idx']]

        if 0 <= obj_idx < self.num_possible_objects:
            # target_object_attrs_str = self.get_target_obj_attrs_str(target_object_str)
            # target_object_name = self.get_target_obj_name_from_str(target_object_str)
            # target_object_name = f"{target_object_attrs_str} {target_object_name}"
            target_object_referents_list = self.create_all_possible_target_obj_referents(
                obj_idx, target_object_str)
        elif self.num_possible_objects <= obj_idx < self.num_obj_idfs:
            target_object_idf = self.get_obj_idf(task_idx)
            extra_obj_idf_scheme_idx = self.get_extra_obj_idf_scheme_idx(task_idx)
            trailing_token = self.extra_obj_idf_instr_trailing_tokens[extra_obj_idf_scheme_idx]
            target_object_referents_list = [f"{target_object_idf} {trailing_token} object"]

        target_container_str = self.container_idfs[indices_map['target_container_idf_idx']]
        target_container_name = self.container_idfs_to_language_name_map[target_container_str]

        if self.deterministic_target_obj_referent:
            # 0th item is default.
            target_object_referents_list = [target_object_referents_list[0]]

        return target_object_referents_list, target_container_name

    def get_objs_and_container_idx_from_task_idx(self, task_idx):
        obj_idx = self.get_obj_idx(task_idx)

        indices_map = super().get_objs_and_container_idx_from_task_idx(task_idx)

        if 0 <= obj_idx < self.num_possible_objects:
            # work already done in super
            pass
        elif self.num_possible_objects <= obj_idx < self.num_obj_idfs:
            indices_map['obj_idf'] = self.get_obj_idf(task_idx)
            objects_matching_idf = self.get_objs_matching_idf(task_idx)
            target_obj_name = np.random.choice(objects_matching_idf)

            # Get chosen_obj_idx and target_obj_idx via helper.
            target_obj_idxs_dict = self.get_obj_idxs_dict_from_obj_name(target_obj_name)
            indices_map.update(target_obj_idxs_dict)
        else:
            raise NotImplementedError

        return indices_map

    def get_target_obj(self, indices_map):
        object_names = tuple(self.possible_objects[indices_map['chosen_obj_idx']])
        target_obj = object_names[indices_map['target_obj_idx']]
        return target_obj

    def get_obj_idxs_dict_from_obj_name(self, obj_name):
        obj_idxs_dict = {}
        for pair_idx, pair in enumerate(self.possible_objects):
            if obj_name in pair:
                obj_idxs_dict["chosen_obj_idx"] = pair_idx
                pair_list = list(pair)
                obj_idxs_dict["target_obj_idx"] = pair_list.index(obj_name)
        return obj_idxs_dict

    def get_obj_idx(self, task_idx):
        return task_idx % self.num_obj_idfs

    def get_extra_obj_idf_scheme_idx(self, task_idx):
        """returns an int in 0, ..., len(obj_name_to_idf_map_list) - 1"""
        obj_idx = self.get_obj_idx(task_idx)
        for scheme_idx, interval in enumerate(
                self.scheme_idx_to_obj_idx_interval_list):
            if obj_idx in range(*interval):
                return scheme_idx
        raise ValueError(f"Unable to get scheme_idx for obj_idx {obj_idx}; not in interval list")

    def get_obj_idf(self, task_idx):
        extra_obj_idf_scheme_idx = self.get_extra_obj_idf_scheme_idx(task_idx)
        starting_obj_idx = self.num_possible_objects + sum(
            self.num_extra_obj_idfs_list[:extra_obj_idf_scheme_idx])
        obj_idf_idx = self.get_obj_idx(task_idx) - starting_obj_idx
        obj_idf = self.extra_obj_idfs_list[extra_obj_idf_scheme_idx][obj_idf_idx]
        return obj_idf

    def create_all_possible_target_obj_referents(self, obj_idx, target_object_str):
        """Ex: if target_object_str is "conic cup", and obj_idx = 0 (currently unused)
        with color extra idf
        "black and white" and shape extra idf "cup", then we have the following:
        attr_list = ["black and white colored", "cup shaped"]
        output: 2^3 - 1 possibilities (3 b/c shape, color, obj_name)
        [
            "black and white colored, cup shaped conic cup", # [1, 1, 1]
            "black and white colored conic cup", # [1, 0, 1]
            "cup shaped conic cup", # [0, 1, 1]
            "black and white colored, cup shaped object" # [1, 1, 0]
            "black and white colored object", # [1, 0, 0]
            "cup shaped object", # [0, 1, 0]
            "conic cup", # [0, 0, 1]
        ]
        """
        def get_target_obj_attrs_list(target_object_str):
            """Returns list of adjective-like descriptions of the target object."""
            attrs_list = []
            for i, obj_name_to_idf_map in enumerate(self.obj_name_to_idf_map_list):
                attr = obj_name_to_idf_map[target_object_str]
                if len(self.extra_obj_idf_instr_trailing_tokens) > 0:
                    attr += f" {self.extra_obj_idf_instr_trailing_tokens[i]}"
                attrs_list.append(attr)
            return attrs_list

        def map_selection_scheme_to_str(selection_scheme, choices):
            """Assumes choices[-1] is the object name"""
            selected = []
            for i in range(len(choices)):
                if selection_scheme[i] == 1:
                    selected.append(choices[i])
                elif i == len(choices) - 1:
                    # Last item. Since we didn't select to have the object name,
                    # keep the string "object" instead.
                    selected.append("object")

            # selected[-1] is either the object name or the string "object"
            selected_as_str = ", ".join(selected[:-1])
            if len(selected_as_str) > 0:
                # Add space only if prefix is non empty string.
                selected_as_str += " "
            selected_as_str += selected[-1]
            return selected_as_str

        dims = len(self.extra_obj_idf_schemes) + 1
        selection_schemes = set(itertools.product([0, 1], repeat=dims)) - set([(0, 0, 0)])
        selection_schemes = list(selection_schemes)
        selection_schemes = sorted(selection_schemes, key=lambda x: -sum(x))
        # gives something like: [
        # (1, 1, 1), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0), (0, 0, 1), (0, 1, 0)]
        # We want all 1's to be in the first spot.
        assert selection_schemes[0] == tuple([1] * dims)

        attrs_list = get_target_obj_attrs_list(target_object_str)
        target_object_name = self.get_target_obj_name_from_str(target_object_str)
        choices = attrs_list + [target_object_name]
        assert len(choices) == dims

        target_obj_referents = []
        for selection_scheme in selection_schemes:
            target_obj_referent = map_selection_scheme_to_str(
                selection_scheme, choices)
            target_obj_referents.append(target_obj_referent)

        return target_obj_referents

    def get_task_lang_dict(self):
        """Each task is a list of strings, not a single string."""
        instructs = []
        target_objs = []
        target_conts = []
        for task_idx in range(self.num_tasks):
            str_list_for_this_task = []
            # This call below may cause reselection
            # of target object name for extra obj idfs
            target_obj_name_list, target_container_name = (
                self.get_target_obj_and_cont_from_task_idx(task_idx))
            for target_object_name in target_obj_name_list:
                task = self.get_task_language(
                    target_object_name, target_container_name)
                str_list_for_this_task.append(task)
            instructs.append(str_list_for_this_task)
            target_objs.append(target_obj_name_list)
            target_conts.append(target_container_name)
        # print("instructs", "\n".join([t[0] for t in instructs]))
        task_lang_dict = dict(
            instructs=instructs,
            target_objs=target_objs,
            target_conts=target_conts,
        )
        return task_lang_dict

    def get_objs_matching_idf(self, task_idx):
        """Returns a list of object_strs matching the current idf"""
        obj_idf = self.get_obj_idf(task_idx)
        extra_obj_idf_scheme_idx = self.get_extra_obj_idf_scheme_idx(task_idx)
        objects_matching_idf = self.idf_to_objs_map_list[extra_obj_idf_scheme_idx][obj_idf]
        return objects_matching_idf


class MultiObjectConcurrentMultiContainerExtraIdentifiersAmbigV3Env(
        MultiObjectConcurrentMultiContainerExtraIdentifiersV3Env):
    """
    2022.10.24, 2022.12.14
    Supports having ambiguous language (for the object-specific tasks)
    """
    def __init__(self, num_ambig_distractors=1, ambig_instructions=True, **kwargs):
        # Given an ambiguous language instruction, how many ambiguous distractors
        # 0 means the language is perfectly non-ambiguous
        self.num_ambig_distractors = num_ambig_distractors
        self.obj_idx_to_ambig_idf_class = None # will be created later
        self.color_to_rgba_map = COLOR_TO_RGBA_MAP
        self.ambig_instructions = ambig_instructions
        # Making this deterministic so that the task language list is deterministic
        self.object_name_to_color_pair_map = OBJECT_NAME_TO_COLOR_PAIR_MAP
        ret = super().__init__(**kwargs)
        assert 0 <= self.num_ambig_distractors <= self.num_objects - 1
        assert self.distractor_obj_hard_mode_prob == 0.0
        assert self.deterministic_target_obj_referent
        return ret

    def get_distractor_obj_names(self, target_obj_name, num_distractor_objs):
        obj_idx = self.get_obj_idx(self.task_idx)
        if num_distractor_objs == 0:
            return []
        elif 0 <= obj_idx < self.num_possible_objects:
            assert num_distractor_objs >= self.num_ambig_distractors
            num_visually_distinct_distractor_objs = (
                num_distractor_objs - self.num_ambig_distractors)
            possible_objs_flattened = [
                item for pair in self.possible_objects for item in pair]
            valid_distractor_obj_names = (
                list(set(possible_objs_flattened) - set([target_obj_name])))
            distractor_obj_names = np.random.choice(
                valid_distractor_obj_names,
                num_visually_distinct_distractor_objs, replace=False)
            return [target_obj_name] + list(distractor_obj_names)
        elif self.num_possible_objects <= obj_idx < self.num_obj_idfs:
            assert num_distractor_objs >= self.num_ambig_distractors
            num_non_ambig_distractor_objs = (
                num_distractor_objs - self.num_ambig_distractors)
            target_obj_idf = self.get_obj_idf(self.task_idx)
            extra_obj_idf_scheme_idx = self.get_extra_obj_idf_scheme_idx(
                self.task_idx)
            obj_name_to_idf_map = self.obj_name_to_idf_map_list[extra_obj_idf_scheme_idx]

            valid_ambig_distractor_obj_names = [
                obj for obj, idf in obj_name_to_idf_map.items()
                if idf == target_obj_idf]
            ambig_distractor_obj_names = np.random.choice(
                valid_ambig_distractor_obj_names, self.num_ambig_distractors,
                replace=False)

            valid_non_ambig_distractor_obj_names = [
                obj for obj, idf in obj_name_to_idf_map.items()
                if idf != target_obj_idf]
            non_ambig_distractor_obj_names = np.random.choice(
                valid_non_ambig_distractor_obj_names,
                num_non_ambig_distractor_objs, replace=False)

            return (list(ambig_distractor_obj_names) +
                    list(non_ambig_distractor_obj_names))
        else:
            raise NotImplementedError

    def get_target_obj_and_cont_from_task_idx(self, task_idx):
        """
        Only used for task language str creation
        """
        indices_map = self.get_objs_and_container_idx_from_task_idx(task_idx)
        obj_idx = self.get_obj_idx(task_idx)

        object_names = tuple(self.possible_objects[indices_map['chosen_obj_idx']])
        target_object_str = object_names[indices_map['target_obj_idx']]

        if 0 <= obj_idx < self.num_possible_objects:
            target_object_name = self.get_target_obj_name_from_str(
                target_object_str)
            target_object_descriptor = (
                self.get_target_obj_directional_descriptor(task_idx))
            target_object_descriptor = (
                self.process_target_obj_directional_descriptor(
                    target_object_descriptor))
            target_object_referent = f"{target_object_descriptor}{target_object_name}"
            target_object_referents_list = [target_object_referent]
        elif self.num_possible_objects <= obj_idx < self.num_obj_idfs:
            target_object_descriptor = (
                self.get_target_obj_directional_descriptor(task_idx))
            target_object_descriptor = (
                self.process_target_obj_directional_descriptor(
                    target_object_descriptor))
            target_object_idf = self.get_obj_idf(task_idx)
            extra_obj_idf_scheme_idx = self.get_extra_obj_idf_scheme_idx(task_idx)
            trailing_token = self.extra_obj_idf_instr_trailing_tokens[extra_obj_idf_scheme_idx]
            target_object_referents_list = [
                f"{target_object_descriptor}{target_object_idf} {trailing_token} object"]

        target_container_str = self.container_idfs[indices_map['target_container_idf_idx']]
        target_container_name = self.container_idfs_to_language_name_map[target_container_str]

        return target_object_referents_list, target_container_name

    def get_target_obj_directional_descriptor(self, task_idx):
        target_container_idf_idx = task_idx // self.num_obj_idfs
        # Need to be deterministic so language is consistent with all demos collected.
        if 0 <= target_container_idf_idx < self.num_containers:
            directional_descriptor = self.extra_container_idfs[
                task_idx % len(self.extra_container_idfs)]
        elif self.num_containers <= target_container_idf_idx < self.num_container_idfs:
            target_container_idf = self.container_idfs[target_container_idf_idx]
            # Ex: target_container_idf = "front"
            container_idf_pair = [
                pair for pair in self.cont_idf_pairs if target_container_idf in pair][0]
            # Ex: container_idf_pair = ["front", "back"]
            directional_descriptor = container_idf_pair[
                task_idx % len(container_idf_pair)]
            # Ex: directional_descriptor is either "front" or "back"
        else:
            raise NotImplementedError
        return directional_descriptor

    def process_target_obj_directional_descriptor(self, target_obj_dir_descr):
        def maybe_add_space(s):
            # Just strips/adds the space when necessary
            if s != "":
                s += " "
            return s

        if self.ambig_instructions:
            # Do not provide directional descriptor if using ambiguous
            # language (default)
            # The below is only for debugging purposes.
            return ""
        else:
            return maybe_add_space(target_obj_dir_descr)

    def create_object_rgbas(self):
        return [None] * self.num_objects
