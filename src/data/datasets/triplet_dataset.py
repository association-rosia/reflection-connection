import os
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

    def __init__(self, wandb_config: dict, image_paths: list, labels: list, processor: dT.RefConfProcessor, train: bool):
        self.wandb_config = wandb_config
        self.dict_path_img = self._make_dict_path_img(labels, image_paths)
        self.set_class_name = set(labels)
        self.processor = processor
        self.train = train
        targets, img_paths = self._remove_targets(labels, image_paths)
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
    def _make_dict_path_img(labels, img_paths):
        dict_path_img = {class_name: [] for class_name in set(labels)}
        for class_name, path_img in zip(labels, img_paths):
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


def get_train_val_split(wandb_config, image_paths, labels):
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        labels, image_paths,
        train_size=0.8,
        random_state=wandb_config['random_state'],
        stratify=labels
    )

    return train_image_paths, val_image_paths, train_labels, val_labels


def get_image_folder(config):
    path = os.path.join(config['path']['data'], 'raw', 'train')
    path = utils.get_notebooks_path(path)

    return path


def get_curated_class_path(config, wandb_config):
    image_folder = get_image_folder(config)
    image_paths, labels = utils.get_paths_labels(image_folder)
    augmented_dataset = utils.load_augmented_dataset(wandb_config)
    augmented_image_paths = [image_dict['image_path'] for image_dict in augmented_dataset]
    augmented_labels = [image_dict['label'] for image_dict in augmented_dataset]
    curated_image_paths = image_paths + augmented_image_paths
    curated_labels = labels + augmented_labels
    
    return curated_image_paths, curated_labels


def make_train_triplet_dataset(config, wandb_config):
    curated_image_paths, curated_labels  = get_curated_class_path(config, wandb_config)
    processor = dT.make_training_processor(config, wandb_config)
    train_image_paths, _, train_labels, _ = get_train_val_split(wandb_config, curated_image_paths, curated_labels)

    return RefConTripletDataset(wandb_config, train_image_paths, train_labels, processor, True)


def make_val_triplet_dataset(config, wandb_config):
    curated_image_paths, curated_labels = get_curated_class_path(config, wandb_config)
    processor = dT.make_training_processor(config, wandb_config)
    _, val_image_paths, _, val_labels = get_train_val_split(wandb_config, curated_image_paths, curated_labels)

    return RefConTripletDataset(wandb_config, val_image_paths, val_labels, processor, False)


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