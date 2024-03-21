import os

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
import wandb.apis.public as wandb_api

from tqdm.autonotebook import tqdm
from typing_extensions import Self

import src.data.datasets.inference as inference_d

import src.models.utils as mutils
from src import utils
 

def load_lightning_model(config: dict, wandb_run: wandb_api.Run | utils.RunDemo, map_location):
    module_lightning = mutils.get_lightning_library(wandb_run.config['model_id'])
    model = module_lightning.get_model(wandb_run.config)
    
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


class InferenceModel(torch.nn.Module):
    def __init__(self,
                 model_id: str,
                 model: torch.nn.Module,
                ) -> None:
        super().__init__()
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
            embeddings = self._vit_torchvision_forward(pixel_values)
        elif 'vit' in self.model_id:
            embeddings = self._vit_transformers_forward(pixel_values)
        
        return embeddings
    
    def _clip_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['image_embeds']

    def _dinov2_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['pooler_output']

    def _vit_torchvision_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)

    def _vit_transformers_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)['pooler_output']


class EmbeddingsBuilder:
    def __init__(self,
                 devices: int | str | list[int] = 0,
                 inference_dtype = torch.float16,
                 batch_size: int = 16,
                 num_workers: int = 16,
                 ) -> None:
        if isinstance(devices, int):
            self.devices = [f'cuda:{devices}']
        elif isinstance(devices, list):
            self.devices = [f'cuda:{device}'for device in devices]
        else:
            self.devices = [devices]
        self.inference_dtype = inference_dtype
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _inference_worker(self,
                          config: dict,
                          wandb_run: wandb_api.Run | utils.RunDemo,
                          device: str,
                          dataset: inference_d.RefConInferenceDataset,
                          embeddings_labels = None):
        
        model = InferenceModel.load_from_wandb_run(config, wandb_run, device)
        model = model.to(dtype=self.inference_dtype)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        embeddings = []
        labels = []
        # Faire l'inférence avec ce DataLoader
        for _, (pixel_values, targets) in enumerate(tqdm(loader)):
            pixel_values = pixel_values.to(device=device, dtype=self.inference_dtype)
            embeddings.append(model(pixel_values).cpu())
            labels.extend(targets)
        
        embeddings = torch.cat(embeddings)
        if embeddings_labels is not None:
            embeddings_labels.append((embeddings, labels))
        else:
            return embeddings, labels

    def _multiprocess_inference(self, config, wandb_run, dataset, output):
        processes = []
        manager = mp.Manager()
        embeddings_labels = manager.list()

        for rank, device in enumerate(self.devices):
            subset_indices = list(range(rank, len(dataset), len(self.devices)))
            subset = Subset(dataset, indices=subset_indices)
            p = mp.Process(target=self._inference_worker, args=(config, wandb_run, device, subset, embeddings_labels))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        embeddings = []
        labels = []
        for embedding, label in embeddings_labels:
            embeddings.append(embedding)
            labels.extend(label)
        
        output.embeddings = torch.cat(embeddings)
        output.labels = labels
    
    def build_embeddings(self, config: dict, wandb_run: wandb_api.Run | utils.RunDemo, dataset: inference_d.RefConInferenceDataset):
        
        if len(self.devices) > 1:
            manager = mp.Manager()
            output = manager.Namespace()
            output.embeddings = None
            output.labels = None
            p = mp.Process(target=self._multiprocess_inference, args=(config, wandb_run, dataset, output))
            p.start()
            p.join()
            return output.embeddings, output.labels
        else:
            return self._inference_worker(config, wandb_run, self.devices[0], dataset)


def _debug():
    config = utils.get_config()
    wandb_run = utils.get_run('omo3q9fq')
    embeddings_builder = EmbeddingsBuilder(devices=[0], batch_size=64, num_workers=32)
    dataset = inference_d.make_iterative_query_inference_dataset(config, wandb_run.config)
    embeddings_builder.build_embeddings(config, wandb_run, dataset)
    emb, lab = embeddings_builder.build_embeddings(config, wandb_run, dataset)
    
    return


if __name__ == '__main__':
    _debug()
