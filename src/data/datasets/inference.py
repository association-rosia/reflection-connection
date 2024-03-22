import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset

import src.data.transforms as dT
from src import utils
import src.data.utils as d_utils


class RefConInferenceDataset(Dataset):
    def __init__(self, wandb_config: dict, image_paths: list, labels: list, processor: dT.RefConfProcessor):
        self.wandb_config = wandb_config
        self.processor = processor
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self._load_image(self.image_paths[idx])

        return image, self.labels[idx]

    def _load_image(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            return self.processor.preprocess_image(img)


def _get_iterative_query_paths_labels(wandb_config, curated_folder):
    augmented_dataset = d_utils.load_augmented_dataset(wandb_config)
    query_paths, query_labels = utils.get_paths_labels(curated_folder)
    query_paths.extend([image_dict['image_path'] for image_dict in augmented_dataset])
    query_labels.extend([image_dict['label'] for image_dict in augmented_dataset])

    return query_paths, query_labels


def _get_submission_paths_labels(folder_path):
    glob_path_png = os.path.join(folder_path, '**', '*.png')
    image_paths = glob(glob_path_png, recursive=True)
    labels = [os.path.basename(image_path) for image_path in image_paths]

    return image_paths, labels


def make_iterative_query_inference_dataset(config, wandb_config):
    curated_folder = os.path.join(config['path']['data'], 'raw', 'train')
    query_paths, query_labels = _get_iterative_query_paths_labels(wandb_config, curated_folder)
    processor = dT.make_eval_processor(config, wandb_config)

    return RefConInferenceDataset(wandb_config, query_paths, query_labels, processor)


def make_iterative_corpus_inference_dataset(config, wandb_config):
    augmented_dataset = d_utils.load_augmented_dataset(wandb_config)
    query_paths = [image_dict['image_path'] for image_dict in augmented_dataset]
    uncurated_folder = os.path.join(config['path']['data'], 'processed', 'pretrain')
    template_path = os.path.join(uncurated_folder, '**', '*.png')
    corpus_paths = glob(template_path, recursive=True)
    corpus_paths = list(set(corpus_paths) - set(query_paths))
    processor = dT.make_eval_processor(config, wandb_config)

    return RefConInferenceDataset(wandb_config, corpus_paths, corpus_paths, processor)


def make_submission_inference_dataset(folder_path, config, wandb_config):
    image_paths, labels = _get_submission_paths_labels(folder_path)
    processor = dT.make_eval_processor(config, wandb_config)

    return RefConInferenceDataset(wandb_config, image_paths, labels, processor)


def make_submission_query_inference_dataset(config, wandb_config):
    query_folder = os.path.join(config['path']['data'], 'raw', 'test', 'query')

    return make_submission_inference_dataset(query_folder, config, wandb_config)


def make_submission_corpus_inference_dataset(config, wandb_config):
    corpus_folder = os.path.join(config['path']['data'], 'raw', 'test', 'image_corpus')

    return make_submission_inference_dataset(corpus_folder, config, wandb_config)


def _debug():
    config = utils.get_config()
    wandb_run = utils.get_run('qlynkj89')
    dataset = make_iterative_corpus_inference_dataset(config, wandb_run.config)
    dataset = make_iterative_query_inference_dataset(config, wandb_run.config)

    return


if __name__ == '__main__':
    _debug()
