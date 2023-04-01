import numpy as np
import subprocess
import time

from video_logger_scripted_policy import *


def get_task_idxs_per_thread(task_idxs, num_threads):
    """
    >>> get_task_idxs_per_thread([0, 1, 2, 3, 4, 5], 2)
    np.array([[0, 1, 2], [3, 4, 5]])
    """
    assert isinstance(task_idxs, list)
    assert len(task_idxs) % num_threads == 0
    task_idxs_per_thread_arr = np.array(
        np.reshape(task_idxs, (num_threads, -1)))
    return task_idxs_per_thread_arr


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("-p", "--num-parallel-threads", type=int, default=1)
    args = parser.parse_args()

    # Remove args that are different per thread
    task_idxs = get_task_idx_list(args)
    delattr(args, "task_idx_intervals")
    delattr(args, "task_idxs")

    # Split task_idxs into $p$ number of threads
    task_idxs_per_thread = get_task_idxs_per_thread(
        task_idxs, args.num_parallel_threads)
    
    command = ["python", "scripts/video_logger_scripted_policy.py"]

    for k, v in args.__dict__.items():
        print(k, v)
        # Don't add certain args
        if k in ["num_parallel_threads"]:
            continue

        k = k.replace("_", "-")

        if isinstance(v, bool):
            # an action="store_true" type of arg
            if v:
                command.append(f"--{k}")
            continue
        elif type(v) in [int, float]:
            v = str(v)
        elif not isinstance(v, str):
            raise ValueError

        command.extend([f"--{k}", v])

    print("command", command)

    subprocesses = []
    for task_idxs in task_idxs_per_thread:
        thread_specific_arg_list = (
            ["--task-idxs"] + [str(x) for x in list(task_idxs)])
        # print("thread_specific_arg_list", thread_specific_arg_list)
        subprocesses.append(
            subprocess.Popen(command + thread_specific_arg_list))
        time.sleep(2)

    exit_codes = [p.wait() for p in subprocesses]
