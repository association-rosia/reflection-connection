from enum import Enum
from typing import overload

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from PIL import Image
from torchvision import transforms


class ProcessorMode(Enum):
    """Processor modes
    Available modes are ``training`` and ``eval``.
    """

    TRAINING = 0
    EVAL = 1


class RefConfProcessor:
    def __init__(self, config: dict, wandb_config: dict, mode: ProcessorMode) -> None:
        self.config = config
        self.wandb_config = wandb_config
        self.mode = mode
        self.random_resized_crop = transforms.RandomResizedCrop(size=self.wandb_config['crop_size'],
                                                                interpolation=tvF.InterpolationMode.BICUBIC)

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
        if self.mode == ProcessorMode.TRAINING:
            return self._preprocess_training_image(image)
        elif self.mode == ProcessorMode.EVAL:
            return self._preprocess_eval_image(image)
        else:
            raise ValueError(f'Mode have an incorrect value')

    def _preprocess_training_image(self, image: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
        image = self._maybe_to_tensor(image)
        image = tvF.adjust_contrast(image, contrast_factor=self.wandb_config['contrast_factor'])
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py#L58
        image = self.random_resized_crop(image)
        image = tvF.normalize(image, mean=self.config['data']['mean'], std=self.config['data']['std'])

        return image

    def _preprocess_eval_image(self, image: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
        image = self._maybe_to_tensor(image)
        image = tvF.adjust_contrast(image, contrast_factor=self.wandb_config['contrast_factor'])
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py#L58
        image = tvF.resize(image, size=self.wandb_config['size'], interpolation=tvF.InterpolationMode.BICUBIC)
        if self.wandb_config.get('crop_size') is not None:
            image = tvF.center_crop(image, output_size=self.wandb_config['crop_size'])
        image = tvF.normalize(image, mean=self.config['data']['mean'], std=self.config['data']['std'])

        return image

    @staticmethod
    def _maybe_to_tensor(pic: torch.Tensor | np.ndarray | Image.Image) -> torch.Tensor:
        if isinstance(pic, torch.Tensor):
            return pic
        return tvF.to_tensor(pic)

    def __call__(self, image: Image.Image | list[Image.Image]) -> torch.Tensor:
        return self.preprocess_image(image)


def make_training_processor(config, wandb_config):
    return RefConfProcessor(config, wandb_config, ProcessorMode.TRAINING)


def make_eval_processor(config, wandb_config):
    return RefConfProcessor(config, wandb_config, ProcessorMode.EVAL)


if __name__ == '__main__':
    from src import utils

    config = utils.get_config()
    wandb_config = utils.load_config('clip.yml')
    make_training_processor(config, wandb_config)
    make_eval_processor(config, wandb_config)
