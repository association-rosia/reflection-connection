from PIL import Image
from torch.utils.data import Dataset

import src.data.transforms as dT
from src import utils
from src.data import utils as d_utils

from sklearn.model_selection import train_test_split

import os
from glob import glob


class RefConViTMAEDataset(Dataset):

    def __init__(self, wandb_config, images_path: list, processor: dT.RefConfProcessor):
        self.wandb_config = wandb_config
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
        item = self._load_image(image_path)

        return item


def make_pretrain_dataset(config, wandb_config, set):
    images_path = d_utils.get_pretraining_images_path(config, set)
    processor = dT.make_pretraining_processor(config, wandb_config)

    return RefConViTMAEDataset(wandb_config, images_path, processor)


def get_fine_tuned_images_path(config, wandb_config):
    train_path = os.path.join(config['path']['data'], 'raw', 'train')
    train_glob = os.path.join(train_path, '**/*.png')
    test_path = os.path.join(config['path']['data'], 'raw', 'test')
    test_glob = os.path.join(test_path, '**/*.png')
    images_path = glob(train_glob, recursive=True) + glob(test_glob, recursive=True)

    train_images_path, val_images_path = train_test_split(
        images_path,
        test_size=0.2,
        random_state=wandb_config['random_state']
    )

    return train_images_path, val_images_path


def make_fine_tuned_dataset(config, wandb_config):
    train_images_path, val_images_path = get_fine_tuned_images_path(config, wandb_config)
    processor = dT.make_pretraining_processor(config, wandb_config)
    train_dataset = RefConViTMAEDataset(wandb_config, train_images_path, processor)
    val_dataset = RefConViTMAEDataset(wandb_config, val_images_path, processor)

    return train_dataset, val_dataset


def _debug():
    from tqdm.autonotebook import tqdm

    config = utils.get_config()
    wandb_config = utils.load_config('pretraining/vitmae.yml')
    dataset = make_fine_tuned_dataset(config, wandb_config, set='val')

    for inputs in tqdm(dataset):
        pass

    pass


if __name__ == '__main__':
    _debug()
