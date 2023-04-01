import argparse
import numpy as np
import os
import os.path as osp
import roboverse
from roboverse.policies import policies
from roboverse.utils import get_timestamp
from roboverse.utils.data_collection_utils import (
    gather_demonstrations_as_hdf5,
    postprocess_traj_dict_for_hdf5, load_env_info,
    maybe_create_data_save_path)
import time
from tqdm import tqdm

EPSILON = 0.1


def add_transition(traj, observation, action, reward, info, agent_info, done,
                   next_observation, img_dim, transpose_image, img_hd_dim):
    def reshape_image(obs, img_dim, transpose_image, img_key):
        if transpose_image:
            obs[img_key] = np.reshape(obs[img_key], (3, img_dim, img_dim))
            obs[img_key] = np.transpose(obs[img_key], [1, 2, 0])
            obs[img_key] = np.uint8(obs[img_key] * 255.)
        else:
            obs[img_key] = np.reshape(
                np.uint8(obs[img_key] * 255.), (img_dim, img_dim, 3))
        return obs

    reshape_image(observation, img_dim, transpose_image, "image")
    reshape_image(next_observation, img_dim, transpose_image, "image")

    if img_hd_dim is not None:
        reshape_image(observation, img_hd_dim, transpose_image, "image_hd")
        reshape_image(next_observation, img_hd_dim, transpose_image, "image_hd")

    traj["observations"].append(observation)
    if "next_observations" in traj:
        traj["next_observations"].append(next_observation)
    traj["actions"].append(action)
    traj["rewards"].append(reward)
    traj["terminals"].append(done)
    traj["agent_infos"].append(agent_info)
    traj["env_infos"].append(info)
    return traj


def collect_one_traj(env, policy, num_timesteps, noise,
                     accept_trajectory_key, transpose_image,
                     use_pretrained_policy=False, lower_gripper_noise=False,
                     task_index=None, fixed_task=False, img_dim=48,
                     img_hd_dim=None, task_language_list=None, gui=False):
    num_steps = -1
    rewards = []
    success = False
    # img_dim = env.observation_img_dim
    policy_reset_kwargs = {}
    observation = env.reset()
    if gui: print("task_index", task_index % env.num_tasks)

    target_object_str = env.target_object
    if hasattr(env, "target_object_idf"):
        target_object_str += f" ({env.target_object_idf})"

    target_container_str = env.target_container
    if hasattr(env, "target_container_idf"):
        target_container_str += f" ({env.target_container_idf})"

    if gui:
        task_str = f"{target_object_str} --> {target_container_str}"
        print(task_str)

        if isinstance(task_language_list, list):
            task_instruction = task_language_list[task_index]
            print(task_instruction)

    if fixed_task:
        policy_reset_kwargs = {"object_to_target": env.target_object}
    policy.reset(**policy_reset_kwargs)

    time.sleep(1)
    traj = dict(
        observations=[],
        actions=[],
        rewards=[],
        terminals=[],
        agent_infos=[],
        env_infos=[],
    )

    if args.save_next_obs:
        traj["next_observations"] = []

    for j in range(num_timesteps):

        observation = env.get_observation()

        if use_pretrained_policy:
            action, agent_info = policy.get_action(observation)
        else:
            action, agent_info = policy.get_action()

        env_action_dim = env.action_space.shape[0]
        if lower_gripper_noise:
            noise_scale = np.array([
                noise if i != policy.gripper_dim else 0.1*noise
                for i in range(env_action_dim)])
        else:
            noise_scale = noise
        action += np.random.normal(scale=noise_scale, size=(env_action_dim,))
        action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
        next_observation, reward, done, info = env.step(action)
        add_transition(traj, observation,  action, reward, info, agent_info,
                       done, next_observation, img_dim, transpose_image,
                       img_hd_dim)
        if info[accept_trajectory_key] and num_steps < 0:
            num_steps = j

        rewards.append(reward)
        if not use_pretrained_policy and (done or agent_info['done']):
            break

    if info[accept_trajectory_key]:
        success = True

    # print("traj['rewards']", traj['rewards'])
    # print("rewards", rewards)
    if gui: print("traj['actions']", np.round(np.array(traj['actions'])[:,6]))

    return traj, success, num_steps


def add_traj_to_be_saved(args, data, traj):
    if args.dset_ext == "npy":
        data.append(traj)
    elif args.dset_ext == "hdf5":
        t1, t2 = str(time.time()).split(".")
        ep_dir = osp.join(args.npz_tmp_dir, f"ep_{t1}_{t2}")
        if not osp.exists(ep_dir):
            os.makedirs(ep_dir)
        state_path = osp.join(ep_dir, f"state_{t1}_{t2}.npz")
        traj = postprocess_traj_dict_for_hdf5(traj)
        save_kwargs = traj
        save_kwargs['env'] = args.env_name
        np.savez(state_path, **save_kwargs)
        # No operations with `data` are needed.
    else:
        raise NotImplementedError
    return data


def scripted_collect(args):
    if args.dset_ext == "hdf5":
        assert args.npz_tmp_dir != ""

    if args.task_idx_intervals == []:
        args.task_indices = list(range(args.num_tasks))
    else:
        task_idx_interval_list = []
        for interval in args.task_idx_intervals:
            interval = tuple([int(x) for x in interval.split("-")])
            assert len(interval) == 2

            if len(task_idx_interval_list) >= 1:
                # Make sure most recently added interval's endpoint is smaller
                # than current interval's startpoint.
                assert task_idx_interval_list[-1][-1] < interval[0]

            task_idx_interval_list.append(interval)

        args.task_indices = []  # to collect_data on
        for interval in task_idx_interval_list:
            start, end = interval
            assert 0 <= start <= end <= args.num_tasks
            args.task_indices.extend(list(range(start, end + 1)))
        print(args.task_indices)

    timestamp = get_timestamp()
    data_save_path = maybe_create_data_save_path(args.save_directory)

    if args.dset_ext == "npy":
        data = []
    elif args.dset_ext == "hdf5":
        data = None
    else:
        raise NotImplementedError

    use_pretrained_policy = False

    if args.pretrained_policy:
        raise NotImplementedError
    else:
        transpose_image = False
        num_tasks_to_collect = len(args.task_indices)

        kwargs = {}
        fixed_task = False
        if args.init_task_idx is not None:
            fixed_task = True
            kwargs = {
                "fixed_task": fixed_task,
                "init_task_idx": args.init_task_idx}
        kwargs.update({
            "observation_img_dim": args.img_dim,
            "observation_img_hd_dim": args.img_hd_dim,
        })
        env = roboverse.make(
            args.env_name, gui=args.gui,
            transpose_image=transpose_image,
            num_tasks=args.num_tasks, **kwargs)
        assert args.accept_trajectory_key in env.get_info().keys(), \
            f"The accept trajectory key must be one of: {env.get_info().keys()}"

    num_success = 0
    num_saved = 0
    num_attempts = 0
    policy = None

    progress_bar = tqdm(total=args.num_trajectories)

    # print(env.get_task_language_list())
    env_task_language_list = env.get_task_lang_dict()['instructs']

    while num_saved < args.num_trajectories:
        num_attempts += 1

        assert args.num_trajectories % num_tasks_to_collect == 0
        num_traj_per_task = args.num_trajectories // num_tasks_to_collect

        if args.init_task_idx is None:
            task_list_idx = num_saved // num_traj_per_task
            task_index = args.task_indices[task_list_idx]
        else:
            task_index = args.init_task_idx

        env.reset_task(task_index)

        policy = policies[args.policy_name](
            env, drop_container_scheme=args.drop_container_scheme,
            pick_object_scheme=args.pick_object_scheme)

        traj, success, num_steps = collect_one_traj(
            env, policy, args.num_timesteps, args.noise,
            args.accept_trajectory_key, transpose_image,
            use_pretrained_policy, args.lower_gripper_noise,
            task_index, fixed_task, args.img_dim,
            args.img_hd_dim, env_task_language_list, args.gui)

        if success or args.save_all:
            if args.gui:
                print("num_timesteps: ", num_steps)
            data = add_traj_to_be_saved(args, data, traj)
            num_success += int(bool(success))
            num_saved += 1
            progress_bar.update(1)

        if args.gui:
            print("success rate: {}".format(num_success/(num_attempts)))

    progress_bar.close()
    print("success rate: {}".format(num_success / (num_attempts)))

    if args.dset_ext == "npy":
        path = osp.join(data_save_path, "scripted_{}_{}.npy".format(
            args.env_name, timestamp))
        print(path)
        np.save(path, data)
    elif args.dset_ext == "hdf5":
        env_info = load_env_info(args)
        path = gather_demonstrations_as_hdf5(
            [args.npz_tmp_dir], data_save_path, env_info)

        # Clean up tmp files
        os.system(f"rm -r {args.npz_tmp_dir}")
        print(f"Removed {args.npz_tmp_dir}")

    return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("-pl", "--policy-name", type=str, required=True)
    parser.add_argument("-pp", "--pretrained-policy", type=str, required=False, default=None)
    parser.add_argument("-a", "--accept-trajectory-key", type=str, required=True)
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("--save-all", action='store_true', default=False)
    parser.add_argument("--gui", action='store_true', default=False)
    parser.add_argument("-d", "--save-directory", type=str, default="")
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("-int", "--task-idx-intervals", nargs="+", type=str, default=[])
    # task-idx-intervals must be in format such as 0-2, 5-9. This means that
    # we collect data on indices [0, 1, 2, 5, 6, 7, 8, 9]
    parser.add_argument("-m", "--num-tasks", type=int, required=True)
    parser.add_argument("--lower-gripper-noise", action='store_true', default=False)
    parser.add_argument("-i", "--init-task-idx", type=int, default=None)
    parser.add_argument("--drop-container-scheme", type=str, default=None)
    parser.add_argument("--pick-object-scheme", type=str, default="random")
    parser.add_argument("--img-dim", type=int, default=48)
    parser.add_argument("--img-hd-dim", type=int, default=None)
    # ^ only looked at for concurrent multicontainer envs.
    parser.add_argument("--dset-ext", type=str, default="npy", choices=["npy", "hdf5"])
    parser.add_argument("--npz-tmp-dir", type=str, default="")
    parser.add_argument("--save-next-obs", action="store_true", default=False)
    args = parser.parse_args()

    scripted_collect(args)
