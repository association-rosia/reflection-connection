import os
from typing_extensions import Self
from tqdm.autonotebook import tqdm

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
import wandb.apis.public as wandb_api

import src.data.datasets.inference_dataset as inf_data

import src.models.utils as mutils
from src import utils
 

def load_lightning_model(config, wandb_run, map_location):
    module_lightning = mutils.get_lightning_library(wandb_run.config)
    model = module_lightning.get_model(wandb_run.config)
    
    kargs = {
        'config': config,
        'wandb_config': wandb_run.config,
        'model': model,
    }

    path_checkpoint = os.path.join(config['path']['models'], f'{wandb_run.name}-{wandb_run.id}.ckpt')
    path_checkpoint = utils.get_notebooks_path(path_checkpoint)
    lightning = module_lightning.RefConLightning.load_from_checkpoint(path_checkpoint, map_location=map_location, **kargs)
    
    return lightning


class InferenceModel(torch.nn.Module):
    def __init__(self,
                 model_id: str,
                 model: torch.nn.Module,
                ) -> None:
        
        self.model_id = model_id
        self.model = model
        self.model.eval()
        
    @classmethod
    def load_from_wandb_run(
        cls,
        config: dict,
        wandb_run: wandb_api.Run | utils.RunDemo,
        map_location) -> Self:
        model = cls._load_model(config, wandb_run, map_location)
        self = cls(wandb_run.config['model_id'], model)
        
        return self
    
    @staticmethod
    def _load_model(config, wandb_run, map_location):
        lightning = load_lightning_model(config, wandb_run, map_location)
        
        return lightning.model

    @torch.inference_mode
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if 'clip' in self.model_id:
            embeddings = self._clip_forward(pixel_values)
        elif 'dinov2' in self.model_id:
            embeddings = self._dinov2_forward(pixel_values)
        elif 'ViT' in self.model_id:
            embeddings = self._vit_forward(pixel_values)
        
        return embeddings
    
    def _clip_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['image_embeds']

    def _dinov2_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['pooler_output']
    
    def _vit_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)


class EmbeddingsBuilder:
    def __init__(self,
                 devices: int | str | list[int] = 0,
                 inference_dtype = torch.float16,
                 batch_size: int = 16,
                 ) -> None:
        if isinstance(devices, int):
            self.devices = [f'cuda:{devices}']
        elif isinstance(devices, list):
            self.devices = [f'cuda:{device}'for device in devices]
        else:
            self.devices = [devices]
        self.inference_dtype = inference_dtype
        self.batch_size = batch_size

    def inference_worker(self, config, wandb_run, device, dataset, embeddings_labels):
        model = InferenceModel.load_from_wandb_run(config, wandb_run, device)
        model = model.to(dtype=self.inference_dtype)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)
        
        # Faire l'inférence avec ce DataLoader
        for _, (pixel_values, targets) in enumerate(tqdm(loader)):
            pixel_values = pixel_values.to(device=device, dtype=self.inference_dtype)
            embeddings = model(pixel_values)
            embeddings_labels.put(embeddings.cpu(), targets)

    def build_embeddings(self, config: dict, wandb_run: wandb_api.Run | utils.RunDemo, dataset: inf_data.RefConInferenceDataset):
        processes = []
        embeddings_labels = mp.Queue()
        
        for rank, device in enumerate(self.devices):
            subset_indices = range(rank, len(dataset), self.devices)
            subset = Subset(dataset, indices=subset_indices)
            p = mp.Process(target=self.inference_worker, args=(config, wandb_run, device, subset, embeddings_labels, self.batch_size))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        embeddings = []
        labels = []
        while not embeddings_labels.empty():
            embedding, label = embeddings_labels.get()
            embeddings.append(embedding)
            labels.append(label)

        return torch.cat(embeddings), torch.cat(labels)


def _debug():
    config = utils.get_config()
    wandb_run = utils.get_run('96t0rkbl')
    embeddings_builder = EmbeddingsBuilder(device=[1, 2])
    dataset = inf_data.make_submission_corpus_inference_dataset(config, wandb_run.config)
    embeddings, labels = embeddings_builder.build_embeddings(config, wandb_run, dataset)

    return


if __name__ == '__main__':
    _debug()