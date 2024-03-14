from enum import Enum
from typing import overload

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from torchvision.transforms import v2


class ProcessorMode(Enum):
    """Processor modes
    Available modes are ``training`` and ``eval``.
    """
    PRETRAINING = -1
    TRAINING = 0
    EVAL = 1


class RefConfProcessor:
    def __init__(self, config: dict, wandb_config: dict, mode: ProcessorMode) -> None:
        self.config = config
        self.wandb_config = wandb_config
        self.mode = mode
        self.random_transforms = self._get_random_transforms()

    def _get_random_transforms(self):
        if self.mode != ProcessorMode.PRETRAINING:
            transforms = v2.Compose([
                v2.RandomResizedCrop(
                    size=self.wandb_config['crop_size'],
                    interpolation=tvF.InterpolationMode.BICUBIC
                ),
                v2.RandomAutocontrast(),
                v2.RandomHorizontalFlip()
            ])
        else:
            transforms = v2.Compose([
                v2.RandomResizedCrop(
                    size=self.wandb_config['crop_size'],
                    interpolation=tvF.InterpolationMode.BICUBIC
                )
            ])

        return transforms

    @overload
    def preprocess_image(self, images: Image.Image) -> torch.Tensor:
        ...

    @overload
    def preprocess_image(self, images: list[Image.Image]) -> torch.Tensor:
        ...

    def preprocess_image(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        if isinstance(images, list):
            return torch.stack([self._preprocess_image(image) for image in images])
        else:
            return self._preprocess_image(images)

    def _preprocess_image(self, image):
        if self.mode == ProcessorMode.PRETRAINING:
            return self._preprocess_pretraining_image(image)
        elif self.mode == ProcessorMode.TRAINING:
            return self._preprocess_training_image(image)
        elif self.mode == ProcessorMode.EVAL:
            return self._preprocess_eval_image(image)
        else:
            raise ValueError(f'Mode have an incorrect value')

    def _preprocess_pretraining_image(self, image: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
        image = tvF.to_image(image)
        image = tvF.to_dtype_image(image, torch.float32, scale=True)
        # image = tvF.adjust_contrast(image, contrast_factor=self.wandb_config['contrast_factor'])
        image = self.random_transforms(image)
        # image = tvF.normalize(image, mean=self.config['data']['mean'], std=self.config['data']['std'])

        return image

    def _preprocess_training_image(self, image: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
        image = tvF.to_image(image)
        image = tvF.to_dtype_image(image, torch.float32, scale=True)
        image = tvF.adjust_contrast(image, contrast_factor=self.wandb_config['contrast_factor'])
        image = self.random_transforms(image)
        # image = tvF.normalize(image, mean=self.config['data']['mean'], std=self.config['data']['std'])

        return image

    def _preprocess_eval_image(self, image: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
        image = tvF.to_image(image)
        image = tvF.to_dtype_image(image, torch.float32, scale=True)
        image = tvF.adjust_contrast(image, contrast_factor=self.wandb_config['contrast_factor'])
        image = tvF.resize(image, size=self.wandb_config['size'], interpolation=tvF.InterpolationMode.BICUBIC)

        if self.wandb_config.get('crop_size') is not None:
            image = tvF.center_crop(image, output_size=self.wandb_config['crop_size'])

        # image = tvF.normalize(image, mean=self.config['data']['mean'], std=self.config['data']['std'])

        return image

    def __call__(self, image: Image.Image | list[Image.Image]) -> torch.Tensor:
        return self.preprocess_image(image)


def make_pretraining_processor(config, wandb_config):
    return RefConfProcessor(config, wandb_config, ProcessorMode.PRETRAINING)


def make_training_processor(config, wandb_config):
    return RefConfProcessor(config, wandb_config, ProcessorMode.TRAINING)


def make_eval_processor(config, wandb_config):
    return RefConfProcessor(config, wandb_config, ProcessorMode.EVAL)


def _debug():
    from src import utils

    config = utils.get_config()
    wandb_config = utils.load_config('clip.yml')
    train_preprocessor = make_training_processor(config, wandb_config)
    eval_preprocessor = make_eval_processor(config, wandb_config)
    img = Image.open('data/raw/train/Boring/abwao.png').convert('RGB')
    train_preprocessor.preprocess_image(img)
    eval_preprocessor.preprocess_image(img)


if __name__ == '__main__':
    _debug()
