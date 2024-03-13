import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image

import src.data.transforms as dT
from src import utils


class RefConTripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, wandb_config: dict, list_class_name: list, list_img_path: list, processor: dT.RefConfProcessor, train: bool):
        self.wandb_config = wandb_config
        self.dict_path_img = self._make_dict_path_img(list_class_name, list_img_path)
        self.set_class_name = set(list_class_name)
        self.processor = processor
        self.train = train
        targets, img_paths = self._remove_targets(list_class_name, list_img_path)
        self.img_paths = img_paths
        self.targets = targets

        if not self.train:
            self.triplets = self._generate_triplets()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.train:
            anchor_img_path = self.img_paths[idx]
            anchor_name = self.targets[idx]
            positive_img_path = self._get_positive_img_path(anchor_name, anchor_img_path)
            negative_img_path = self._get_negative_img_path(anchor_name)
        else:
            anchor_img_path = self.triplets[idx][0]
            positive_img_path = self.triplets[idx][1]
            negative_img_path = self.triplets[idx][2]
            
        anchor_img = self._load_image(anchor_img_path)
        positive_img = self._load_image(positive_img_path)
        negative_img = self._load_image(negative_img_path)

        return anchor_img, positive_img, negative_img
    
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
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            return self.processor.preprocess_image(img)
    
    @staticmethod
    def _get_random_state(random_state = None):
        if random_state is None:
            return np.random
        else: 
            return random_state

    def _get_positive_img_path(self, anchor_name, anchor_img_path, random_state = None):
        random_state = self._get_random_state(random_state)
        positive_img_path = anchor_img_path
        while positive_img_path == anchor_img_path:
            positive_img_path = random_state.choice(self.dict_path_img[anchor_name])
        
        return positive_img_path
    
    def _get_negative_img_path(self, anchor_name, random_state = None):
        random_state = self._get_random_state(random_state)
        possible_classes = np.fromiter(self.set_class_name - {anchor_name}, dtype=object)
        negative_name = random_state.choice(possible_classes)
        
        return random_state.choice(self.dict_path_img[negative_name])
    
    def _generate_triplets(self):
        random_state = np.random.RandomState(self.wandb_config['random_state'])
        triplets = []
        for anchor_img_path, anchor_name in zip(self.img_paths, self.targets):
            positive_img_path = self._get_positive_img_path(anchor_name, anchor_img_path, random_state)
            negative_img_path = self._get_negative_img_path(anchor_name, random_state)
            triplets.append((anchor_img_path, positive_img_path, negative_img_path))

        return triplets


def get_class_path(dir_path):
    list_class_name = []
    list_img_path = []
    for class_name in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_path, img_name)
                    list_class_name.append(class_name)
                    list_img_path.append(img_path)
    
    return list_class_name, list_img_path


def get_iterative_class_path(wandb_config, iterative_file):
    list_class_name = []
    list_img_path = []
    
    curated_data = utils.load_curated_dataset()
    
    return list_class_name, list_img_path


def get_train_val_split(wandb_config, list_class_name, list_img_path):
    train_class_name, val_class_name, train_path_img, val_path_img = train_test_split(
        list_class_name, list_img_path,
        train_size=0.8,
        random_state=wandb_config['random_state'],
        stratify=list_class_name
    )

    return train_class_name, val_class_name, train_path_img, val_path_img


def get_image_folder(config):
    path = os.path.join(config['path']['data'], 'raw', 'train')
    path = utils.get_notebooks_path(path)

    return path


def get_curated_class_path(config, wandb_config):
    # TODO: Faire en sorte que si le iterative file n'existe pas 
    # alors ça va chercher le train d'origine sinon ça se réfère à l'iterative file
    image_folder = get_image_folder(config)
    list_class_name, list_img_path = get_class_path(image_folder)
    iterative_file = get_iterative_file(config, wandb_config)
    list_iterative_class_name, list_iterative_img_path = get_iterative_class_path(wandb_config, iterative_file)
    curated_class_name = list_class_name + list_iterative_class_name
    curated_img_path = list_img_path + list_iterative_img_path
    
    return curated_class_name, curated_img_path


def make_train_triplet_dataset(config, wandb_config):
    curated_class_name, curated_img_path = get_curated_class_path(config, wandb_config)
    processor = dT.make_training_processor(config, wandb_config)
    train_class_name, _, train_path_img, _ = get_train_val_split(wandb_config, curated_class_name, curated_img_path)

    return RefConTripletDataset(wandb_config, train_class_name, train_path_img, processor, True)


def make_val_triplet_dataset(config, wandb_config):
    curated_class_name, curated_img_path = get_curated_class_path(config, wandb_config)
    processor = dT.make_training_processor(config, wandb_config)
    _, val_class_name, _, val_path_img = get_train_val_split(wandb_config, curated_class_name, curated_img_path)

    return RefConTripletDataset(wandb_config, val_class_name, val_path_img, processor, False)


def _debug():
    from tqdm.autonotebook import tqdm

    config = utils.get_config()
    wandb_config = utils.load_config('clip.yml')
    val_dataset = make_val_triplet_dataset(config, wandb_config)
    train_dataset = make_train_triplet_dataset(config, wandb_config)

    for inputs in tqdm(val_dataset):
        pass

    for inputs in tqdm(train_dataset):
        pass

    pass


if __name__ == '__main__':
    _debug()