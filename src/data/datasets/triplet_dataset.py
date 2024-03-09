import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

import src.data.transforms as dT
from src import utils


class RefConTripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, wandb_config: dict, dataset: ImageFolder, transform: dT.RefConfProcessor, indices: list[int],
                 train: bool):
        self.wandb_config = wandb_config
        dataset.transform = transform
        self.indices = indices
        self.labels_set = set(dataset.class_to_idx.values())
        self.targets = np.asarray(dataset.targets)[self.indices]
        self.train = train
        self.label_to_indices = {label: np.where(np.asarray(self.targets) == label)[0]
                                 for label in self.labels_set}
        self.subset_dataset = Subset(dataset, self.indices)

        if not self.train:
            self.triplets = self.generate_triplets()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.train:
            anchor_img, anchor_label = self.subset_dataset[idx]
            positive_idx = idx
            while positive_idx == idx:
                positive_idx = np.random.choice(self.label_to_indices[anchor_label])
            negative_label = np.random.choice(list(self.labels_set - set([anchor_label])))
            negative_idx = np.random.choice(self.label_to_indices[negative_label])
            positive_img, positive_label = self.subset_dataset[positive_idx]
            negative_img, negative_label = self.subset_dataset[negative_idx]
        else:
            anchor_img, anchor_label = self.subset_dataset[self.triplets[idx][0]]
            positive_img, positive_label = self.subset_dataset[self.triplets[idx][1]]
            negative_img, negative_label = self.subset_dataset[self.triplets[idx][2]]

        assert anchor_label == positive_label
        assert anchor_label != negative_label

        return anchor_img, positive_img, negative_img

    def generate_triplets(self):
        random_state = np.random.RandomState(self.wandb_config['random_state'])
        triplets = []
        for idx in range(len(self.indices)):
            positive_idx = random_state.choice(self.label_to_indices[self.targets[idx]])
            negative_label = random_state.choice(list(self.labels_set - set([self.targets[idx]])))
            negative_idx = random_state.choice(self.label_to_indices[negative_label])
            triplets.append((idx, positive_idx, negative_idx))

        return triplets


def get_train_val_indices(wandb_config, dataset: ImageFolder):
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        train_size=0.8,
        random_state=wandb_config['random_state'],
        stratify=dataset.targets
    )

    return train_indices, val_indices


def make_image_folder_dataset(config):
    path = os.path.join(config['path']['data'], 'raw', 'train')
    path = utils.get_notebooks_path(path)

    return ImageFolder(path)


def make_train_triplet_dataset(config, wandb_config):
    dataset = make_image_folder_dataset(config)
    processor = dT.make_training_processor(config, wandb_config)
    train_indices, _ = get_train_val_indices(wandb_config, dataset)

    return RefConTripletDataset(wandb_config, dataset, processor, train_indices, True)


def make_val_triplet_dataset(config, wandb_config):
    dataset = make_image_folder_dataset(config)
    processor = dT.make_eval_processor(config, wandb_config)
    _, val_indices = get_train_val_indices(wandb_config, dataset)

    return RefConTripletDataset(wandb_config, dataset, processor, val_indices, False)


if __name__ == '__main__':
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
