import os
from glob import glob
from importlib import import_module
from typing import overload

import torch
import wandb.apis.public as wandb_api
from PIL import Image
from tqdm.autonotebook import tqdm
from typing_extensions import Self

import src.data.transforms as dT
from src import utils


def _import_module_lightning(model_id):
    if 'clip' in model_id:
        return import_module('src.models.training.clip.lightning')
    elif 'dinov2' in model_id:
        return import_module('src.models.training.dinov2.lightning')
    elif 'ViT' in model_id:
        return import_module('src.models.training.vit.torchvision.lightning')
    elif 'vit' in model_id:
        return import_module('src.models.training.vit.transformers.lightning')


def load_lightning_model(config, wandb_run, map_location):
    model_id = wandb_run.config['model_id']
    module_lightning = _import_module_lightning(model_id)

    if 'clip' in model_id or 'dinov2' in model_id or 'ViT' in model_id:
        model = module_lightning.get_model(wandb_run.config)
    else:
        model = module_lightning.get_model(config, wandb_run.config)

    kwargs = {
        'config': config,
        'wandb_config': wandb_run.config,
        'model': model,
    }

    path_checkpoint = os.path.join(config['path']['models'], f'{wandb_run.name}-{wandb_run.id}.ckpt')
    path_checkpoint = utils.get_notebooks_path(path_checkpoint)
    lightning = module_lightning.RefConLightning.load_from_checkpoint(path_checkpoint, map_location=map_location,
                                                                      **kwargs)

    return lightning


class InferenceModel:
    def __init__(self,
                 config: dict,
                 wandb_config: dict,
                 model: torch.nn.Module,
                 device: str
                 ) -> None:

        self.config = config
        self.wandb_config = wandb_config
        self.model = model
        self.device = device
        self.dtype = torch.float32
        self.processor = dT.make_eval_processor(config, self.wandb_config)
        self.model.to(dtype=self.dtype, device=device)
        self.model.eval()

    def to(self, device):
        self.device = device
        self.model.to(device=device)

    @classmethod
    def load_from_wandb_run(
            cls,
            config: dict,
            wandb_run: wandb_api.Run | utils.RunDemo,
            map_location) -> Self:
        model = cls._load_model(config, wandb_run, map_location)
        self = cls(config, wandb_run.config, model, map_location)

        return self

    @staticmethod
    def _load_model(config, wandb_run, map_location):
        lightning = load_lightning_model(config, wandb_run, map_location)

        return lightning.model

    @torch.inference_mode
    def forward(self, images: list[Image.Image] | Image.Image) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]
        pixel_values = self.processor.preprocess_image(images)
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        if 'clip' in self.wandb_config['model_id']:
            embeddings = self._clip_forward(pixel_values)
        elif 'dinov2' in self.wandb_config['model_id']:
            embeddings = self._dinov2_forward(pixel_values)
        elif 'ViT' in self.wandb_config['model_id']:
            embeddings = self._vit_torchvision_forward(pixel_values)
        elif 'vit' in self.wandb_config['model_id']:
            embeddings = self._vit_transformers_forward(pixel_values)

        return embeddings.squeeze(dim=0).cpu()

    def _clip_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['image_embeds']

    def _dinov2_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['pooler_output']

    def _vit_torchvision_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)

    def _vit_transformers_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['pooler_output']

    def __call__(self, images: list[Image.Image] | Image.Image) -> torch.Tensor:
        return self.forward(images)


class EmbeddingsBuilder:
    def __init__(self,
                 device: int | str = 0,
                 return_names: bool = True,
                 ) -> None:
        if isinstance(device, int):
            self.device = f'cuda:{device}'
        else:
            self.device = device
        self.return_names = return_names

    def _load_model(self, config, wandb_run):
        return InferenceModel.load_from_wandb_run(config, wandb_run, self.device)

    def _get_model(self, model=None, config=None, wandb_run=None):
        if model is None:
            model = self._load_model(config, wandb_run)
        else:
            model.to(device=self.device)

        return model

    @staticmethod
    def _make_list_paths(folder_path):
        glob_path = os.path.join(folder_path, '**', '*.png')

        return glob(glob_path, recursive=True)

    @staticmethod
    def _load_image(image_path):
        with open(image_path, 'rb') as f:
            return Image.open(f).convert(mode='RGB')

    @overload
    def build_embeddings(self, model: InferenceModel, folder_path: str, return_names: bool = None):
        ...

    @overload
    def build_embeddings(self, model: InferenceModel, list_paths: list[str], return_names: bool = None):
        ...

    @overload
    def build_embeddings(self, config: dict, wandb_run: wandb_api.Run | utils.RunDemo, folder_path: str,
                         return_names: bool = None):
        ...

    @overload
    def build_embeddings(self, config: dict, wandb_run: wandb_api.Run | utils.RunDemo, list_paths: list[str],
                         return_names: bool = None):
        ...

    def build_embeddings(self, model=None, config=None, wandb_run=None, folder_path=None, list_paths=None,
                         return_names=False):
        model = self._get_model(model, config, wandb_run)

        # Si la liste des images n'est pas fournie, récupère la liste de toutes les images du folder
        if list_paths is None:
            list_paths = self._make_list_paths(folder_path)

        embeddings = []
        names = []
        for img_path in tqdm(list_paths):
            img = self._load_image(img_path)
            embeddings.append(model(img))
            names.append(os.path.basename(img_path))
        embeddings = torch.stack(embeddings)

        if (return_names is None and self.return_names) or return_names:
            return embeddings, names
        else:
            return embeddings


def _debug():
    config = utils.get_config()
    wandb_run = utils.get_run('zgv4h86p')
    model = InferenceModel.load_from_wandb_run(config, wandb_run, 'cpu')
    embeddings_builder = EmbeddingsBuilder(device=0, return_names=True)
    folder_path = os.path.join(config['path']['data'], 'raw', 'train')
    embeddings_builder.build_embeddings(model=model, folder_path=folder_path)

    del model, embeddings_builder
    torch.cuda.empty_cache()


if __name__ == '__main__':
    _debug()
