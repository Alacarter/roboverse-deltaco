from roboverse.utils.data_collection_utils import (
    load_env_info, concat_hdf5,
    maybe_create_data_save_path)
import argparse
from glob import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-names", nargs='+', type=str, required=True)
    parser.add_argument("-p", "--hdfs", nargs="+", type=str, default=[])
    parser.add_argument("-d", "--save-directory", type=str, required=True)
    parser.add_argument("--remove-next-obs", action="store_true", default=False)
    args = parser.parse_args()

    if len(args.hdfs) >= 1:
        hdf5_paths = args.hdfs
        out_dir = args.save_directory
        if len(args.hdfs) == 1:
            assert args.remove_next_obs
            input("You only have one dataset to concatenate, are you sure? CTRL+C to quit.")
    elif len(args.hdfs) == 0:
        # Search for *.hdf5 files in args.save_directory, concatenate them all.
        data_save_path = maybe_create_data_save_path(args.save_directory)
        thread_outpaths = os.path.join(data_save_path, "*.hdf5")
        hdf5_paths = list(glob(thread_outpaths))
        out_dir = data_save_path

    print("hdf5_paths", hdf5_paths)
    env_info = load_env_info(args)
    keys_to_remove = []
    if args.remove_next_obs:
        keys_to_remove.append("next_observations")
    concat_hdf5(hdf5_paths, out_dir, env_info, args.env_names[0], keys_to_remove)

    # Clean up tmp files
    tmp_dir = f"{args.save_directory}/tmp/"
    if os.path.exists(tmp_dir):
        os.system(f"rm -r {tmp_dir}")
        print(f"Removed {tmp_dir}")
