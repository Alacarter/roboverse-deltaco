import argparse
import os

import h5py
import nexusformat.nexus as nx
import numpy as np

from roboverse.utils import get_timestamp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to hdf5 file")
    args = parser.parse_args()

    print("nxload-ing hdf5...")
    f = nx.nxload(args.path)
    print("done loading nxload.")
    print(str(f.tree))
    fname = f"hdf5_tree_{get_timestamp()}.txt"
    # with open(fname, "w") as fout:
    #     fout.write(str(f.tree))

    with h5py.File(args.path, mode='r') as e:
        print(e['data'].attrs['env_info'])
        for task_idx in e['data'].keys():
            task_idx_traj_ids = sorted(list(set([int(traj_id[5:]) for traj_id in e[f'data/{task_idx}'].keys()])))
            # print(f"Task {task_idx} trajs total: {len(task_idx_traj_ids)}:", task_idx_traj_ids)
            traj_obs_sums = []
            for traj_id in task_idx_traj_ids:
                import ipdb; ipdb.set_trace()
                traj_obs = e[f'data/{task_idx}/demo_{traj_id}/observations/image']
                # traj_next_obs = e[f'data/{task_idx}/demo_{traj_id}/next_observations/image']
                traj_obs_sum = np.sum(traj_obs)
                # traj_next_obs_sum = np.sum(traj_next_obs)
                traj_obs_sums.append(traj_obs_sum)
                # traj_obs_sums.append(traj_next_obs_sum) # throwing into the same list for analysis
            print(f"task {task_idx}: # distinct obs and next_obs trajs: {len(set(traj_obs_sums))}")
        import ipdb; ipdb.set_trace()
