import os

import numpy as np
from PIL import Image

from torch.utils.data import Dataset

import src.data.transforms as dT
from src import utils
from src.data import utils as d_utils


class RefConOnlineMiningTripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self,
                 wandb_config: dict,
                 list_class_name: list,
                 list_img_path: list,
                 processor: dT.RefConfProcessor
                 ):
        self.wandb_config = wandb_config
        self.list_class_name = list_class_name
        self.processor = processor
        _, self.img_paths = self._remove_targets(list_class_name, list_img_path)

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _make_dict_path_img(list_class_name, img_paths):
        dict_path_img = {class_name: [] for class_name in set(list_class_name)}
        for class_name, path_img in zip(list_class_name, img_paths):
            dict_path_img[class_name].append(path_img)

        return {k: np.asarray(v) for k, v in dict_path_img.items()}

    def _remove_targets(self, targets, img_paths):
        targets = np.asarray(targets)
        img_paths = np.asarray(img_paths)
        mask = np.isin(targets, self.wandb_config['exclude_anchor'], invert=True)

        return targets[mask], img_paths[mask]

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return self.processor.preprocess_image(img)

    def _img_path_to_label(self, img_path):
        class_name = img_path.split('/')[-2]
        label = sorted(set(self.list_class_name)).index(class_name)

        return label

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self._load_image(img_path)
        label = self._img_path_to_label(img_path)

        return image, label


def make_train_triplet_dataset(config, wandb_config):
    image_folder = d_utils.get_image_folder(config)
    list_class_name, list_img_path = d_utils.get_class_path(image_folder)
    processor = dT.make_training_processor(config, wandb_config)
    train_class_name, _, train_path_img, _ = d_utils.get_train_val_split(wandb_config, list_class_name, list_img_path)

    return RefConOnlineMiningTripletDataset(wandb_config, train_class_name, train_path_img, processor)


def make_val_triplet_dataset(config, wandb_config):
    image_folder = d_utils.get_image_folder(config)
    list_class_name, list_img_path = d_utils.get_class_path(image_folder)
    processor = dT.make_training_processor(config, wandb_config)
    _, val_class_name, _, val_path_img = d_utils.get_train_val_split(wandb_config, list_class_name, list_img_path)

    return RefConOnlineMiningTripletDataset(wandb_config, val_class_name, val_path_img, processor)


def _debug():
    from tqdm.auto import tqdm

    config = utils.get_config()
    wandb_config = utils.load_config('training/dinov2.yml')
    val_dataset = make_val_triplet_dataset(config, wandb_config)
    train_dataset = make_train_triplet_dataset(config, wandb_config)

    for inputs in tqdm(val_dataset):
        pass

    for inputs in tqdm(train_dataset):
        pass

    pass


if __name__ == '__main__':
    _debug()
