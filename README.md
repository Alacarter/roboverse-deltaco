# minibullet-ut

## Introduction
This codebase contains the environments and benchmark for the DeL-TaCo paper:

**Using Both Demonstrations and Language Instructions to Efficiently Learn Robotic Tasks** <br />
Albert Yu, Raymond J. Mooney <br />
ICLR (International Conference on Learning Representations), 2023 <br />
[Web](https://deltaco-robot.github.io/) | [PDF](https://openreview.net/pdf?id=4u42KCQxCn8) <br />


This codebase builds on the environments from the previously released repo, [roboverse](https://github.com/avisingh599/roboverse), which was released with the COG paper.

## Setup
### Conda env creation
```
conda env create -n deltaco python=3.6
pip install -r requirements.txt
pip install -e .
python setup.py develop
```

### Clone bullet-objects
If needed, update the filepaths stored in the variables `SHAPENET_ASSET_PATH` and `BASE_ASSET_PATH` in `roboverse/bullet/object_utils.py`.

## Scripted data collection

### <a name="scripted-gui"></a> To visualize an environment/scripted policy
```
python scripts/scripted_collect.py -e Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 -pl pickplace -a place_success_target_obj_target_container -n 2 -t 30 -d [output path] --task-idx-intervals 0-1 --num-tasks 300 --pick-object-scheme target --drop-container-scheme target --dset-ext hdf5 --npz-tmp-dir [empty temp dir path] --gui
```

### To collect data on multiple threads in parallel
See our released datasets page for downloads to our open-sourced datasets.

The following scripts only collect successful trajectories, where a success is defined by the `-a` env info param being true. To collect all trajectories regardless of their success, add the flag `--save-all`.

#### To recollect T198
```
python scripts/scripted_collect_parallel.py -e Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 -pl pickplace -a place_success_target_obj_target_container -n 178200 -p [num threads to use; must be a factor of (178200 / 198) = 900] -t 30 -d [output path] --task-idx-intervals 0-23 36-44 50-73 86-94 100-123 136-144 150-173 186-194 200-223 236-244 250-273 286-294 --num-tasks 300 --pick-object-scheme target --drop-container-scheme target --dset-ext hdf5
```

#### To recollect T48
```
python scripts/scripted_collect_parallel.py -e Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 -pl pickplace -a place_success_target_obj_target_container -n 43200 -p [num threads to use; must be a factor of (43200 / 48) = 900] -t 30 -d [output path] --task-idx-intervals 24-31 74-81 124-131 174-181 224-231 274-281 --num-tasks 300 --pick-object-scheme target --drop-container-scheme target --dset-ext hdf5
```

#### To recollect T54
```
python scripts/scripted_collect_parallel.py -e Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 -pl pickplace -a place_success_target_obj_target_container -n 43200 -p [num threads to use; must be a factor of (48600 / 54) = 900] -t 30 -d [output path] --task-idx-intervals 32-35 45-49 82-85 95-99 132-135 145-149 182-185 195-199 232-235 245-249 282-285 295-299 --num-tasks 300 --pick-object-scheme target --drop-container-scheme target --dset-ext hdf5
```

#### To recollect E48 + E54
```
python scripts/scripted_collect_parallel.py -e Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 -pl pickplace -a place_success_target_obj_target_container -n 1020 -p 10 -t 30 -d [output path] --task-idx-intervals 24-35 45-49 74-85 95-99 124-135 145-149 174-185 195-199 224-235 245-249 274-285 295-299 --num-tasks 300 --pick-object-scheme target --drop-container-scheme target --dset-ext npy
```


## Environments
### Current environment logic
To see how and where the environment used in our paper (`Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0`) is defined, check the following files:
 - Registration: `roboverse/envs/registration.py`, in the dictionary with `id`: `Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0`
 - Env class definition: The registration entry defines the entry point as `roboverse.envs.widow250_pickplace:Widow250PickPlaceMultiObjectConcurrentMultiContainerExtraIdentifiersV3Env`, which means that the env class is the long name after the `:` and the filepath is `roboverse/envs/widow250_pickplace.py`. From this, the inheritance tree of the environment can be traced back to `Widow250Env`. Important functions:
   - `step(...)`: Executes an input action
   - `reset(...)`: Places environment objects and containers when each episode begins
   - `get_observation(...)`: Information about the state that is storable in the dataset. Only the image and robot state are given to the policy during training.
   - `_reset_container_and_obj_location_ranges(...)`: Called upon each reset. Sets `self.container_name_to_position_map`, a dictionary mapping the container name to its xyz position upon each reset. It also sets the ranges for where an object can be placed.
   - `get_cont_and_obj_quadrants(...)`: Chooses the container and object coordinates upon each reset.
   - `get_task_lang_dict(...)`: Returns a dictionary with 3 keys: `instructs`, which maps to a list of instructions where the instruction at index `i` corresponds to the instruction of task ID `i`.
   - `get_distractor_obj_names(...)`: Returns a list of distractor objects to appear in the env upon each reset, along with the target object.

### Adding environments
1. Create a new class from the environment that inherits from an existing class, such as `Widow250Env` in `roboverse/envs/widow250.py`, `Widow250PickPlaceEnv` in `roboverse/envs/widow250_pickplace.py`, etc.
2. Add an entry in `roboverse/envs/registration.py` by creating a new dictionary element in the `ENVIRONMENT_SPECS`. The dictionary must contain three entries:
  - `id`: the string name for your new env. This is the arg you pass in on command line to `scripted_collect.py` and during training.
  - `entry_point`: the path to the env class. For instance, an env class called `Widow250PickPlaceEnv` defined in the file `roboverse/envs/widow250_pickplace.py` has the `entry_point` `roboverse.envs.widow250_pickplace:Widow250PickPlaceEnv`.
  - `kwargs`: a dictionary of parameters to be passed into the init method of the env class you defined in (1).
3. You can verify what your environment looks like by running a `scripted_collect.py` command with the `--gui` flag and a scripted policy name after the `-pl` flag as [shown above](#scripted-gui).


## Scripted Policies
### Current scripted policy
The scripted pick and place policy, used during data collection, is defined in `PickPlace` in `roboverse/policies/pick_place.py`.
 - `get_action(...)` returns the action for the scripted policy to take, which it computes based on the current end effector position and the target object location.
 - `reset(...)` is called when an episode is done and the state variables of the scripted policy need to be reset or recomputed.

### Adding new scripted policies
1. Define a new class with methods `get_action(...)` and `reset(...)`.
2. Import the class into `roboverse/policies/__init__.py` and define a key-value pair in the `policies` dictionary where the key is the name of the policy that you will pass in on the command line, and the value is the class you defined.

## Miscellaneous scripts
### Keyboard control
```
python scripts/keyboard_control.py -e Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0
```

### Concatenating Datasets
```
python scripts/concatenate_datasets_hdf5.py -e Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 -p [paths of hdf5 datasets] -d [output folder path]
```

### Creating videos from a scripted policy
```
python scripts/video_logger_scripted_policy.py -e Widow250PickPlaceGRFBLRObjCSRndDistractorRndTrayQuad-v0 -pl pickplace -a place_success_target_obj_target_container -t 30 -d scripts/debug -n 2 --save-mode mp4 --task-idx-intervals 1-2 --num-tasks 300
```

### Citation
```
@inproceedings{yu:2023,
  title={Using Both Demonstrations and Language Instructions to Efficiently Learn Robotic Tasks},
  author={Albert Yu and Raymond J. Mooney},
  booktitle={Proceedings of the International Conference on Learning Representations, 2023},
  year={2023},
}
```
