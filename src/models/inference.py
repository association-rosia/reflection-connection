import os
from typing import overload
from importlib import import_module
from tqdm.autonotebook import tqdm
from glob import glob

import torch
import numpy as np
from PIL import Image
import wandb.apis.public as wandb_api

import src.data.transforms as dT
from src import utils


def _import_module_lightning(model_id):
    if 'clip' in model_id:
        return import_module('src.models.clip.lightning')
    elif 'dinov2' in model_id:
        return import_module('src.models.dinov2.lightning')
    

def load_lightning_model(config, wandb_run, map_location):
    module_lightning = _import_module_lightning(wandb_run.config['model_id'])
    model = module_lightning.get_model(wandb_run.config)
    
    kargs = {
        'config': config,
        'wandb_config': wandb_run.config,
        'model': model,
    }

    path_checkpoint = os.path.join(config['path']['models']['root'], f'{wandb_run.name}-{wandb_run.id}.ckpt')
    path_checkpoint = utils.get_notebooks_path(path_checkpoint)
    lightning = module_lightning.RefConLightning.load_from_checkpoint(path_checkpoint, map_location=map_location, **kargs)
    
    return lightning


class InferenceModel(torch.nn.Module):
    def __init__(self,
                 config: dict,
                 wandb_config: dict,
                 model: torch.nn.Module,
                ) -> None:
        super().__init__()
        
        self.config = config
        self.wandb_config = wandb_config
        self.model = model
        self.processor = dT.make_eval_processor(config, self.wandb_config)
        self.to(dtype=torch.float16, device=self.model.device)
        self.eval()
        
    @classmethod
    def load_from_wandb_run(
        cls,
        config: dict,
        wandb_run: wandb_api.Run | utils.RunDemo,
        cuda_idx):
        
        map_location = f'cuda:{cuda_idx}'
        model = cls._load_model(config, wandb_run, map_location)
        self = cls(config, wandb_run.config, model)
        self.to(dtype=torch.float16, device=self.model.device)
        
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

        pixel_values = pixel_values.to(device=self.model.device)
        if 'clip' in self.wandb_config['model_id']:
            embeddings = self._clip_forward(pixel_values)
        elif 'dinov2' in self.wandb_config['model_id']:
            embeddings = self._dinov2_forward(pixel_values)
        
        return embeddings.squeeze(dim=0).cpu()
    
    def _clip_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['image_embeds']

    def _dinov2_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['pooler_output']


class EmbeddingsBuilder:
    def __init__(self,
                 cuda_idx: int = 0,
                 return_labels: bool = True,
                 ) -> None:
        self.cuda_idx = cuda_idx
        self.return_labels = return_labels

    def _load_model(self, config, wandb_run):
        return InferenceModel.load_from_wandb_run(config, wandb_run, self.cuda_idx)
    
    def _get_model(self, model = None, config = None, wandb_run = None):
        if model is None:
            return self._load_model(config, wandb_run)
        else:
            return model.to(device=f'cuda:{self.cuda_idx}')
    
    @staticmethod
    def _make_list_paths(folder_path):
        glob_path = os.path.join(folder_path, '**', '*.png')
        
        return glob(glob_path, recursive=True)
    
    @staticmethod
    def _load_image(image_path):
        with open(image_path, 'rb') as f:
            return Image.open(f).convert(mode='RGB')

    @overload
    def build_embeddings(self, model: InferenceModel, folder_path: str, return_names: bool): ...
    @overload
    def build_embeddings(self, model: InferenceModel, list_paths: list[str], return_names: bool): ...
    @overload
    def build_embeddings(self, config: dict, wandb_run: wandb_api.Run | utils.RunDemo, folder_path: str, return_names: bool): ...
    @overload
    def build_embeddings(self, config: dict, wandb_run: wandb_api.Run | utils.RunDemo, list_paths: list[str], return_names: bool): ...
    def build_embeddings(self, model = None, config = None, wandb_run = None, folder_path = None, list_paths = None, return_names = False):
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
        
        del model
        torch.cuda.empty_cache()
        
        if return_names:
            return embeddings, names
        else:
            return embeddings


def _debug():
    config = utils.get_config()
    wandb_run = utils.get_run('sxa0zzzr')
    model = InferenceModel.load_from_wandb_run(config, wandb_run, 0)
    embeddings_builder = EmbeddingsBuilder(cuda_idx=0, return_labels=True)
    embeddings_builder.build_embeddings(model=model, folder_path=config['path']['data']['raw']['train'])

    del model, embeddings_builder
    torch.cuda.empty_cache()

if __name__ == '__main__':
    _debug()