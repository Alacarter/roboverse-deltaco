import argparse
import datetime
from glob import glob
import os
import subprocess
import threading
import time

from roboverse.utils import get_timestamp
from roboverse.utils.data_collection_utils import (
    load_env_info, concat_hdf5,
    maybe_create_data_save_path)
from scripts.scripted_collect import scripted_collect


def get_data_save_directory(args):
    data_save_directory = args.data_save_directory

    data_save_directory += '_{}'.format(args.env_name)

    if args.num_trajectories > 1000:
        data_save_directory += '_{}K'.format(int(args.num_trajectories/1000))
    else:
        data_save_directory += '_{}'.format(args.num_trajectories)

    if args.pretrained_policy:
        data_save_directory += "_pretrained_policy"

    if args.save_all:
        data_save_directory += '_save_all'

    data_save_directory += '_noise_{}'.format(args.noise)
    data_save_directory += '_{}'.format(get_timestamp())

    return data_save_directory


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("-pl", "--policy-name", type=str, required=True)
    parser.add_argument("-pp", "--pretrained-policy", type=str, required=False, default=None)
    parser.add_argument("-a", "--accept-trajectory-key", type=str, required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("-d", "--data-save-directory", type=str, required=True)
    parser.add_argument("--save-all", action='store_true', default=False)
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--lower-gripper-noise", action='store_true', default=False)
    parser.add_argument("-m", "--num-tasks", type=int, required=True)
    parser.add_argument("-int", "--task-idx-intervals", nargs="+", type=str, default=[])
    parser.add_argument("-i", "--init-task-idx", type=int, default=None)
    parser.add_argument("--drop-container-scheme", type=str, default=None)
    parser.add_argument("--pick-object-scheme", type=str, default="random")
    parser.add_argument("--img-dim", type=int, default=48)
    parser.add_argument("--img-hd-dim", type=int, default=None)
    parser.add_argument("--dset-ext", type=str, default="npy", choices=["npy", "hdf5"])
    parser.add_argument("--save-next-obs", action='store_true', default=False)
    parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()

    num_trajectories_per_thread = int(
        args.num_trajectories / args.num_parallel_threads)
    if args.num_trajectories % args.num_parallel_threads != 0:
        num_trajectories_per_thread += 1

    save_directory = get_data_save_directory(args)
    args.save_directory = save_directory

    script_name = "scripted_collect.py"

    if args.pretrained_policy is not None:
        policy_flag_args = ["--pretrained-policy={}".format(args.pretrained_policy)]
        if args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    elif args.policy_name != "":
        policy_flag_args = [f"-pl={args.policy_name}"]
    else:
        raise NotImplementedError

    command = (['python',
               'scripts/{}'.format(script_name)] +
               policy_flag_args +
               [f'-a={args.accept_trajectory_key}'] +
               [f'-e={args.env_name}'] +
               ['-n {}'.format(num_trajectories_per_thread)] +
               [f'-t={args.num_timesteps}'] +
               [f'-m={args.num_tasks}'] +
               ['-d={}'.format(save_directory)] +
               ['--pick-object-scheme', args.pick_object_scheme] +
               [f'--img-dim={args.img_dim}'] +
               [f'--dset-ext={args.dset_ext}'])

    if args.save_all:
        command.append('--save-all')
    if args.save_next_obs:
        command.append('--save-next-obs')
    if args.lower_gripper_noise:
        command.append('--lower-gripper-noise')
    if args.init_task_idx is not None:
        command.append('-i {}'.format(args.init_task_idx))
    if args.drop_container_scheme is not None:
        command.extend(['--drop-container-scheme', args.drop_container_scheme])
    if args.img_hd_dim is not None:
        command.extend(['--img-hd-dim={}'.format(args.img_hd_dim)])
    if args.task_idx_intervals != []:
        command.extend(['-int'] + args.task_idx_intervals)

    subprocesses = []
    for i in range(args.num_parallel_threads):
        thread_specific_arg_list = []
        if args.dset_ext == "hdf5":
            npz_tmp_dir = f"{save_directory}/tmp/{get_timestamp()}"
            thread_specific_arg_list.extend(["--npz-tmp-dir", npz_tmp_dir])
        elif args.dset_ext == "npy":
            pass
        print("thread_specific_arg_list", thread_specific_arg_list)
        subprocesses.append(subprocess.Popen(command + thread_specific_arg_list))
        time.sleep(2)

    exit_codes = [p.wait() for p in subprocesses]

    if args.dset_ext == "npy":
        merge_command = ['python',
                     'scripts/combine_trajectories.py',
                     '-d{}'.format(save_directory)]
        subprocess.call(merge_command)
    elif args.dset_ext == "hdf5":
        data_save_path = maybe_create_data_save_path(args.save_directory)
        thread_outpaths = os.path.join(data_save_path, "*.hdf5")
        # There should only be 1 *.npz file under the ep_directory
        thread_outpaths = list(glob(thread_outpaths))
        print("thread_outpaths", thread_outpaths)
        env_info = load_env_info(args)
        concat_hdf5(thread_outpaths, data_save_path, env_info, args.env_name)

        # Clean up tmp files
        os.system(f"rm -r {save_directory}/tmp/")
        print(f"Removed {save_directory}/tmp/")
    else:
        raise NotImplementedError
