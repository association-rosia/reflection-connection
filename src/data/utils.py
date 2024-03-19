import os
from glob import glob


def get_images_path(config, set):
    if set == 'train':
        pretrain_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
        pretrain_glob = os.path.join(pretrain_path, '**/*.png')
        images_path = glob(pretrain_glob, recursive=True)
    elif set == 'val':
        train_path = os.path.join(config['path']['data'], 'raw', 'train')
        train_glob = os.path.join(train_path, '**/*.png')
        test_path = os.path.join(config['path']['data'], 'raw', 'test')
        test_glob = os.path.join(test_path, '**/*.png')
        images_path = glob(train_glob, recursive=True) + glob(test_glob, recursive=True)
    else:
        raise ValueError(f'Unknown set: {set}')

    return images_path