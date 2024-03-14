import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset

import src.data.transforms as dT
from src import utils

import matplotlib.pyplot as plt


class RefCoPretrainDataset(Dataset):

    def __init__(self, images_path: list, processor: dT.RefConfProcessor):
        self.images_path = images_path
        self.processor = processor

    def __len__(self):
        return len(self.images_path)

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return self.processor.preprocess_image(img)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]

        image = self._load_image(image_path)
        plt.imshow(image.permute(1, 2, 0))
        plt.show()

        image = self._load_image(image_path)
        plt.imshow(image.permute(1, 2, 0))
        plt.show()

        return


def get_images_path(config):
    pretrain_train_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
    pretrain_train_glob = os.path.join(pretrain_train_path, '**/*.png')
    images_path = glob(pretrain_train_glob, recursive=True)

    return images_path


def make_petrain_dataset(config, wandb_config):
    images_path = get_images_path(config)
    processor = dT.make_pretraining_processor(config, wandb_config)

    return RefCoPretrainDataset(images_path, processor)


def _debug():
    from tqdm.autonotebook import tqdm

    config = utils.get_config()
    wandb_config = utils.load_config('pretraining.yml')
    pretrain_dataset = make_petrain_dataset(config, wandb_config)

    for inputs in tqdm(pretrain_dataset):
        pass

    pass


if __name__ == '__main__':
    _debug()
