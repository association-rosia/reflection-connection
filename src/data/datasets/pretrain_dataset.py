import os
from glob import glob
import torch

from PIL import Image
from torch.utils.data import Dataset

import src.data.transforms as dT
from src import utils


class RefCoPretrainDataset(Dataset):

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
        dino_student_inputs = self._load_image(image_path)
        dino_teacher_inputs = self._load_image(image_path)

        ibot_inputs = self._load_image(image_path)
        ibot_bool_masked_pos = torch.randint(0, 2, (self.wandb_config['num_patches'],)).bool()

        item = {
            'dino_student_inputs': dino_student_inputs,
            'dino_teacher_inputs': dino_teacher_inputs,
            'ibot_inputs': ibot_inputs,
            'ibot_bool_masked_pos': ibot_bool_masked_pos
        }

        return item


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


def make_petrain_dataset(config, wandb_config, set):
    images_path = get_images_path(config, set)
    processor = dT.make_pretraining_processor(config, wandb_config)

    return RefCoPretrainDataset(wandb_config, images_path, processor)


def _debug():
    from tqdm.autonotebook import tqdm

    config = utils.get_config()
    wandb_config = utils.load_config('pretrain.yml')
    pretrain_dataset = make_petrain_dataset(config, wandb_config)

    for inputs in tqdm(pretrain_dataset):
        pass

    pass


if __name__ == '__main__':
    _debug()
