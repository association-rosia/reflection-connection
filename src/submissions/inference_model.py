from PIL import Image
import os
import torch
import numpy as np
from importlib import import_module
import wandb.apis.public as wandb_api

import src.data.make_dataset as md
from src import utils


class RefConInferenceModel(torch.nn.Module):
    def __init__(self,
                 config,
                 wandb_run: wandb_api.Run | utils.RunDemo,
                 cuda_idx: int = 0
                ) -> None:
        super().__init__()
        
        self.config = config
        self.wandb_run = wandb_run
        self.cuda_device = f'cuda:{cuda_idx}'
        self.make_lightning = self._import_make_lightning()
        self.model = self._load_model()
        self.processor = md.RefConfProcessor(self.config, self.wandb_run.config)
        self.to(device=self.cuda_device, dtype=torch.float16)
        self.eval()
        
        
    def _import_make_lightning(self):
        if 'clip' in self.wandb_run.config['model_id']:
            make_lightning = import_module('src.models.clip.make_lightning')
        
        return make_lightning
    
    def _load_model(self):
        base_dataset = md.get_base_dataset(self.config)
        train_indices, val_indices = md.get_train_val_indices(self.wandb_run.config, base_dataset)
        model = self.make_lightning.get_model(self.wandb_run.config)
        
        args = {
            'config': self.config,
            'wandb_config': self.wandb_run.config,
            'model': model,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'dataset': base_dataset
        }

        path_checkpoint = os.path.join(self.config['path']['models']['root'], f'{self.wandb_run.name}-{self.wandb_run.id}.ckpt')
        path_checkpoint = utils.get_notebooks_path(path_checkpoint)
        lightning = self.make_lightning.RefConLightning.load_from_checkpoint(path_checkpoint, map_location=self.cuda_device, args=args)
        model = lightning.model
        del lightning

        return model

    @torch.inference_mode
    def forward(self, images: list[Image.Image] | np.ndarray) -> torch.Tensor:
        if isinstance(images, np.ndarray):
            pixel_values = self.processor(images)
        else:
            images = [images] if isinstance(images, Image.Image) else images
            list_pixel_values = [self.processor(image) for image in images]
            pixel_values = torch.stack(list_pixel_values)

        pixel_values = pixel_values.to(device=self.model.device)
        if 'clip' in self.wandb_run.config['model_id']:
            embeddings = self.clip_forward(pixel_values)
        
        return embeddings.squeeze(dim=0).cpu()
    
    def clip_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(pixel_values)['image_embeds']

        return embeddings

if __name__ == '__main__':
    config = utils.get_config()
    wandb_run = utils.get_run('2khs9u4f')
    model = RefConInferenceModel(config, wandb_run, 0)
    image = Image.open('/home/external-rosia/RosIA/reflection-connection/data/raw/train/Boring/abwao.png')
    image = image.convert("RGB")
    outputs = model(images=image)

    del model
    torch.cuda.empty_cache()