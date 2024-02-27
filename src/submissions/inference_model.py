from PIL import Image
import os
import torch
import numpy as np
from importlib import import_module
import wandb.apis.public as wandb_api

# import src.data.datasets.triplet_dataset as td
import src.data.transforms as dT
from src import utils


class RefConInferenceModel(torch.nn.Module):
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
        model = _load_model(config, wandb_run, map_location)
        self = cls(config, wandb_run.config, model)
        self.to(dtype=torch.float16, device=self.model.device)
        
        return self


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


def _import_module_lightning(model_id):
    if 'clip' in model_id:
        return import_module('src.models.clip.lightning')
    elif 'dinov2' in model_id:
        return import_module('src.models.dinov2.lightning')


def _load_model(config, wandb_run, map_location):
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
    model = lightning.model
    del lightning

    return model


def _debug():
    config = utils.get_config()
    wandb_run = utils.get_run('sxa0zzzr')
    model = RefConInferenceModel.load_from_wandb_run(config, wandb_run, 0)
    image = Image.open('/home/external-rosia/RosIA/reflection-connection/data/raw/train/Boring/abwao.png')
    image = image.convert("RGB")
    outputs = model(images=image)

    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    _debug()