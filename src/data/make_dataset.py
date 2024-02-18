from torch.utils.data import Subset
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as tvF
from tqdm import tqdm
import torch
from src import utils


class RefConDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, config, wandb_config, dataset: ImageFolder, indices: list[int], train: bool):
        self.config = config
        self.wandb_config = wandb_config
        self.dataset = dataset
        self.indices = indices
        self.labels_set = set(self.dataset.class_to_idx.values())
        self.targets = np.asarray(dataset.targets)[self.indices]
        self.train = train
        self.label_to_indices = {label: np.where(np.asarray(self.targets) == label)[0]
                                     for label in self.labels_set}
        self.subset = Subset(self.dataset, self.indices)
        
        if not self.train:
            self.triplets = self.generate_triplets()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if self.train:
            img1, label1 = self.subset[idx]
            positive_idx = idx
            while positive_idx == idx:
                positive_idx = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_idx = np.random.choice(self.label_to_indices[negative_label])
            img2, _ = self.subset[positive_idx]
            img3, _ = self.subset[negative_idx]
        else:
            img1, _ = self.subset[self.triplets[idx][0]]
            img2, _ = self.subset[self.triplets[idx][1]]
            img3, _ = self.subset[self.triplets[idx][2]]
        
        img1 = self.preprocess_image(img1)
        img2 = self.preprocess_image(img2)
        img3 = self.preprocess_image(img3)

        return img1, img2, img3
    
    def generate_triplets(self):
        random_state = np.random.RandomState(self.wandb_config['random_state'])
        triplets = []
        for idx in range(len(self.indices)):
            positive_idx = random_state.choice(self.label_to_indices[self.targets[idx]])
            negative_label = random_state.choice(list(self.labels_set - set([self.targets[idx]])))
            negative_idx = random_state.choice(self.label_to_indices[negative_label])
            triplets.append((idx, positive_idx, negative_idx))

        return triplets

    def preprocess_image(self, image):
        image = tvF.to_tensor(image)
        image = tvF.adjust_contrast(image, contrast_factor=self.wandb_config['contrast_factor'])
        image = tvF.resize(image, size=self.wandb_config['size'], interpolation=tvF.InterpolationMode.BILINEAR)
        image = tvF.normalize(image, mean=self.config['data']['mean'], std=self.config['data']['std'])
        
        return image


def get_train_val_indices(wandb_config, dataset: ImageFolder):
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        train_size=0.8,
        random_state=wandb_config['random_state'],
        stratify=dataset.targets
    )
    
    return train_indices, val_indices


def compute_image_mean_std():
    config = utils.get_config()
    dataset = ImageFolder(config['path']['data']['raw']['train'])
    # All images have not the same shape
    sum = torch.tensor(0.)
    count = torch.tensor(0.)
    print('\nCompute mean:')
    for img, _ in tqdm(dataset):
        img = tvF.to_tensor(img)
        img = tvF.adjust_contrast(img, contrast_factor=25)
        sum += torch.sum(img)
        count += img.nelement()

    mean = sum / count

    sum = torch.tensor(0.)
    print('\nCompute std:')
    for img, _ in tqdm(dataset):
        img = tvF.to_tensor(img)
        img = tvF.adjust_contrast(img, contrast_factor=25)
        sum += torch.sum((img - mean) ** 2)

    std = torch.sqrt(sum / count)
    
    print('mean', mean)
    print('std', std)


def get_base_dataset(config):
    path = config['path']['data']['raw']['train']
    path = utils.get_notebooks_path(path)
    
    return ImageFolder(path)


if __name__ == '__main__':
    config = utils.get_config()
    wandb_config = utils.load_config('clip.yml')
    dataset = get_base_dataset(config)
    # compute_image_mean_std()
    train_indices, val_indices = get_train_val_indices(wandb_config, dataset)
    train_refcon_dataset = RefConDataset(config, wandb_config, dataset, train_indices, True)
    val_refcon_dataset = RefConDataset(config, wandb_config, dataset, train_indices, False)
    
    pass