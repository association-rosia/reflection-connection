import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import src.data.transforms as dT
from src import utils
from src.data import utils as d_utils


class RefConTripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self,
                 wandb_config: dict,
                 image_paths: list,
                 labels: list,
                 processor: dT.RefConfProcessor,
                 train: bool
                 ):
        self.wandb_config = wandb_config
        self.dict_path_img = self._make_dict_path_img(image_paths, labels)
        self.set_class_name = set(labels)
        self.processor = processor
        self.train = train
        self.image_paths, self.targets = self._remove_targets(image_paths, labels)
        
        if not self.train:
            self.triplets = self._generate_triplets()

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def _make_dict_path_img(image_paths, labels):
        dict_image_paths = {label: [] for label in set(labels)}
        for image_path, label in zip(image_paths, labels):
            dict_image_paths[label].append(image_path)

        return {k: np.asarray(v) for k, v in dict_image_paths.items()}

    def _remove_targets(self, image_paths, labels):
        targets = np.asarray(labels)
        image_paths = np.asarray(image_paths)
        mask = np.isin(targets, self.wandb_config['exclude_anchor'], invert=True)

        return image_paths[mask], targets[mask]

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return self.processor.preprocess_image(img)

    @staticmethod
    def _get_random_state(random_state=None):
        if random_state is None:
            return np.random
        else:
            return random_state

    def _get_positive_image_path(self, anchor_name, anchor_img_path, random_state=None):
        random_state = self._get_random_state(random_state)
        positive_img_path = anchor_img_path
        while positive_img_path == anchor_img_path:
            positive_img_path = random_state.choice(self.dict_path_img[anchor_name])

        return positive_img_path

    def _get_negative_image_path(self, anchor_name, random_state=None):
        random_state = self._get_random_state(random_state)
        possible_classes = np.fromiter(self.set_class_name - {anchor_name}, dtype=object)
        negative_name = random_state.choice(possible_classes)

        return random_state.choice(self.dict_path_img[negative_name])

    def _generate_triplets(self):
        random_state = np.random.RandomState(self.wandb_config['random_state'])
        triplets = []
        for anchor_img_path, anchor_name in zip(self.image_paths, self.targets):
            positive_img_path = self._get_positive_image_path(anchor_name, anchor_img_path, random_state)
            negative_img_path = self._get_negative_image_path(anchor_name, random_state)
            triplets.append((anchor_img_path, positive_img_path, negative_img_path))

        return triplets

    def __getitem__(self, idx):
        if self.train:
            anchor_img_path = self.image_paths[idx]
            anchor_name = self.targets[idx]
            positive_img_path = self._get_positive_image_path(anchor_name, anchor_img_path)
            negative_img_path = self._get_negative_image_path(anchor_name)
        else:
            anchor_img_path = self.triplets[idx][0]
            positive_img_path = self.triplets[idx][1]
            negative_img_path = self.triplets[idx][2]

        anchor_img = self._load_image(anchor_img_path)
        positive_img = self._load_image(positive_img_path)
        negative_img = self._load_image(negative_img_path)

        return anchor_img, positive_img, negative_img


def make_train_triplet_dataset(config, wandb_config):
    curated_image_paths, curated_labels = d_utils.get_curated_class_path(config)
    processor = dT.make_training_processor(config, wandb_config)
    train_image_paths, _, train_labels, _ = d_utils.get_train_val_split(wandb_config, curated_image_paths, curated_labels)
    augmented_dataset = d_utils.load_augmented_dataset(wandb_config)
    augmented_image_paths = [image_dict['image_path'] for image_dict in augmented_dataset]
    augmented_labels = [image_dict['label'] for image_dict in augmented_dataset]
    train_image_paths.extend(augmented_image_paths)
    train_labels.extend(augmented_labels)
    
    return RefConTripletDataset(wandb_config, train_image_paths, train_labels, processor, True)


def make_val_triplet_dataset(config, wandb_config):
    curated_image_paths, curated_labels = d_utils.get_curated_class_path(config)
    processor = dT.make_training_processor(config, wandb_config)
    _, val_image_paths, _, val_labels = d_utils.get_train_val_split(wandb_config, curated_image_paths, curated_labels)

    return RefConTripletDataset(wandb_config, val_image_paths, val_labels, processor, False)


def _debug():
    from tqdm.autonotebook import tqdm

    config = utils.get_config()
    wandb_config = utils.load_config('training/dinov2.yml')
    val_dataset = make_val_triplet_dataset(config, wandb_config)
    train_dataset = make_train_triplet_dataset(config, wandb_config)

    return


if __name__ == '__main__':
    _debug()
