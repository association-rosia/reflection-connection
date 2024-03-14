from torch.utils.data import Dataset
from src import utils
import os
from glob import glob


class RefCoPretrainDataset(Dataset):

    def __init__(self):
        return

    def __len__(self):
        return

    def __getitem__(self, idx):
        return

:
def get_images_path(config):
    pretrain_train_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
    pretrain_train_glob = os.path.join(pretrain_train_path, '**/*.png')
    images_path = glob(pretrain_train_glob, recursive=True)

    return images_path

def make_petrain_dataset(config):
    images_path = get_images_path(config)

    return


def _debug():
    from tqdm.autonotebook import tqdm

    config = utils.get_config()
    pretrain_dataset = make_petrain_dataset(config)

    for inputs in tqdm(pretrain_dataset):
        pass

    pass


if __name__ == '__main__':
    _debug()