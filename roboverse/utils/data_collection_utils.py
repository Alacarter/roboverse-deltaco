from collections import Counter
import datetime
from glob import glob
import h5py
import json
import numpy as np
import os
import os.path as osp
from tqdm import tqdm

from roboverse.utils import get_timestamp


def postprocess_traj_dict_for_hdf5(traj):
    assert isinstance(traj, dict)

    def flatten_env_infos(traj):
        flattened_env_infos = {}
        for env_info_key in traj['env_infos'][0]:
            if type(traj['env_infos'][0][env_info_key]) in [int, bool]:
                flattened_env_infos[env_info_key] = np.array(
                    [traj['env_infos'][i][env_info_key]
                        for i in range(len(traj['env_infos']))])
        return flattened_env_infos

    def flatten_obs(traj_obs):
        flattened_obs = {}
        for key in traj_obs[0].keys():
            flattened_obs[key] = np.array(
                [traj_obs[i][key] for i in range(len(traj_obs))])
        return flattened_obs

    orig_traj_keys = list(traj.keys())
    flattened_keys = set(['actions', 'rewards', 'terminals'])
    keys_to_remove = set(['agent_infos'])
    keys_to_flatten = sorted(
        list(set(traj.keys()) - flattened_keys - keys_to_remove))

    for key in orig_traj_keys:
        if key in keys_to_remove:
            traj.pop(key, None)
        elif key in keys_to_flatten:
            if key in ["observations", "next_observations"]:
                traj[key] = flatten_obs(traj[key])
            elif key == "env_infos":
                traj['env_infos'] = flatten_env_infos(traj)
            else:
                raise NotImplementedError
        else:
            assert key in flattened_keys

    return traj


def create_hdf5_datasets_from_dict(grp, dic):
    for k, v in dic.items():
        grp.create_dataset(k, data=v)


def gather_demonstrations_as_hdf5(dir_list, out_dir, env_info):
    """
    This function is largely taken from:
    https://github.com/ARISE-Initiative/robosuite/blob/1b825f11a937f5c18f2ac167af8ab084275fc625/robosuite/scripts/collect_human_demonstrations.py#L83

    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        dir_list (list of strs): each element is a Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """
    print("dir_list", dir_list)
    timestamp = get_timestamp()
    hdf5_path = os.path.join(out_dir, f"{timestamp}.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0

    task_idx_to_grp_map = dict()
    task_idx_to_num_trajs_counter = Counter()

    for directory in dir_list:
        for ep_directory in os.listdir(directory):

            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
            # There should only be 1 *.npz file under the ep_directory
            assert len(glob(state_paths)) == 1
            state_file = glob(state_paths)[0]

            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            task_idx = dic["env_infos"][()]["task_idx"][0]
            # print("task_idx", task_idx)
            task_idx_traj_num = task_idx_to_num_trajs_counter[task_idx]
            if task_idx_traj_num == 0:
                task_grp = grp.create_group(f"{task_idx}")
                task_idx_to_grp_map[task_idx] = task_grp
            else:
                task_grp = task_idx_to_grp_map[task_idx]

            ep_data_grp = task_grp.create_group(
                "demo_{}".format(task_idx_traj_num))

            # write datasets for all items in the trajectory.

            obs_ep_grp = ep_data_grp.create_group("observations")
            create_hdf5_datasets_from_dict(obs_ep_grp, dic['observations'][()])

            ep_data_grp.create_dataset("actions", data=dic['actions'])
            ep_data_grp.create_dataset("rewards", data=dic['rewards'])

            if "next_observations" in dic:
                nobs_ep_grp = ep_data_grp.create_group("next_observations")
                create_hdf5_datasets_from_dict(
                    nobs_ep_grp, dic['next_observations'][()])

            ep_data_grp.create_dataset("terminals", data=dic['terminals'])

            env_infos_ep_grp = ep_data_grp.create_group("env_infos")
            create_hdf5_datasets_from_dict(
                env_infos_ep_grp, dic['env_infos'][()])

            ep_data_grp.create_dataset("env", data=env_name)

            n_sample = ep_data_grp["actions"].shape[0]
            ep_data_grp.attrs["num_samples"] = n_sample

            num_eps += 1
            task_idx_to_num_trajs_counter[task_idx] += 1

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    print("saved to", hdf5_path)
    f.close()
    return hdf5_path


def concat_hdf5(hdf5_list, out_dir, env_info, env_name, keys_to_remove=[]):
    timestamp = get_timestamp()
    out_path = os.path.join(out_dir, f"scripted_{env_name}_{timestamp}.hdf5")
    f_out = h5py.File(out_path, mode='w')
    grp = f_out.create_group("data")

    task_idx_to_num_eps_map = Counter()
    env_args = None
    for h5name in tqdm(hdf5_list):
        h5fr = h5py.File(h5name, 'r')
        if "env_args" in h5fr['data'].attrs:
            env_args = h5fr['data'].attrs['env_args']
        for task_idx in h5fr['data'].keys():
            if task_idx not in f_out['data'].keys():
                task_idx_grp = grp.create_group(task_idx)
            else:
                task_idx_grp = f_out[f'data/{task_idx}']
            task_idx = int(task_idx)
            for demo_id in h5fr[f'data/{task_idx}'].keys():
                task_idx_traj_num = task_idx_to_num_eps_map[task_idx]
                new_name = f"demo_{task_idx_traj_num}"
                h5fr.copy(
                    f"data/{task_idx}/{demo_id}", task_idx_grp, name=new_name)
                for key_to_remove in keys_to_remove:
                    if key_to_remove in task_idx_grp[new_name].keys():
                        del task_idx_grp[f"{new_name}/{key_to_remove}"]
                task_idx_to_num_eps_map[task_idx] += 1

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["env"] = env_name
    if env_args is not None:
        grp.attrs["env_args"] = env_args
    grp.attrs["env_info"] = env_info
    # grp.attrs["orig_hdf5_list"] = hdf5_list

    print("saved to", out_path)
    f_out.close()

    return out_path


def load_env_info(args):
    config = {"env_name": args.env_name}
    env_info = json.dumps(config)
    return env_info


def maybe_create_data_save_path(save_directory):
    data_save_path = osp.join(__file__, "../..", "data", save_directory)
    data_save_path = osp.abspath(data_save_path)
    if not osp.exists(data_save_path):
        os.makedirs(data_save_path)
    return data_save_path
