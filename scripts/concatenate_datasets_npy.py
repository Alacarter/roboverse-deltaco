import argparse
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', required=True)
    parser.add_argument('-s', '--input-sizes', nargs='+', required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    input_files = args.input
    input_sizes = args.input_sizes
    assert len(input_files) == len(input_sizes)

    all_files = []
    print('loading..')
    for f, s in tqdm(zip(input_files, input_sizes)):
        print(f, "size to keep", s)
        data = np.load(f, allow_pickle=True)
        if s != "all":
            try:
                num_trajs_to_keep = int(s)
                data = data[:num_trajs_to_keep]
            except:
                raise NotImplementedError
        all_files.append(data)
    print('concatenating..')
    all_data = np.concatenate(all_files, axis=0)
    print('saving data of shape {}'.format(all_data.shape))
    np.save(args.output, all_data)
