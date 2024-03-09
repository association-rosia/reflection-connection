import os
from glob import glob

import numpy as np
from tqdm.auto import tqdm

import src.utils as utils


def main():
    config = utils.get_config()
    data_pretrain_path = os.path.join(config['path']['data'], 'raw', 'pretrain')
    data_pretrain_glob = os.path.join(data_pretrain_path, '**/*.npy')

    pbar = tqdm(glob(data_pretrain_glob, recursive=True))
    for volume_path in pbar:
        try:
            volume = np.load(volume_path)
            # print(f'OK - {volume_path}')
        except:
            pbar.write(f'ERROR - {volume_path}')


if __name__ == "__main__":
    main()
