import copy
import json
import multiprocessing as mp
import os

import numpy as np
import torch
import wandb
import wandb.apis.public as wandb_api
from torch.utils.data import Subset

import src.data.datasets.inference as inference_d
import src.models.utils as m_utils
from src import utils
from src.models.inference import EmbeddingsBuilder
from src.models.retriever import FaissRetriever


def main():
    config = utils.get_config()
    iterative_config = utils.load_config('fine_tuning/iterative.yml')
    curated_folder = os.path.join(config['path']['data'], 'raw', 'train')

    iterative_trainer = IterativeTrainer(
        config,
        iterative_config,
        curated_folder
    )

    iterative_trainer.fit()


class IterativeTrainer:
    def __init__(self,
                 config,
                 iterative_config,
                 curated_folder) -> None:
        self.config = config
        self.iterative_config = iterative_config
        self.curated_folder = curated_folder
        self.path_iterative_data = os.path.join(self.config['path']['data'], 'processed', 'train')

    def fit(self):
        manager = mp.Manager()
        fit_dict = manager.dict()
        iterative_data = self.iterative_config.get('iterative_data', 'None')
        fit_dict['iterative_data'] = iterative_data if iterative_data != 'None' else None
        for _ in range(self.iterative_config['iterations']):
            p = mp.Process(target=self._train_model, args=(fit_dict,))
            p.start()
            p.join()
            wandb_run = utils.get_run(fit_dict['wandb_id'])
            
            p = mp.Process(target=self._create_next_iterative_dataset, args=(wandb_run, fit_dict))
            p.start()
            p.join()

    def _train_model(self, fit_dict):
        utils.init_wandb(self.iterative_config['model_config'], self.iterative_config['sub_config'])
        wandb.config.update({
            'iterative_data': fit_dict['iterative_data'],
            'curated_threshold': self.iterative_config['curated_threshold'],
            'duplicate_threshold': self.iterative_config['duplicate_threshold'],
            'devices': [2],
            # 'devices': self.iterative_config['devices'],
            'dry': self.iterative_config['dry'],
            'checkpoint': self.iterative_config['checkpoint'],
        }, allow_val_change=True)
        fit_dict['wandb_id'] = copy.deepcopy(wandb.run.id)
        trainer = m_utils.get_trainer(self.config)
        lightning = m_utils.get_lightning(self.config, wandb.config)
        trainer.fit(model=lightning)
        del trainer, lightning
        torch.cuda.empty_cache()
        wandb.finish()

    def _create_next_iterative_dataset(self, wandb_run: utils.RunDemo | wandb_api.Run, fit_dict):
        embeddings_builder = EmbeddingsBuilder(devices=self.iterative_config['devices'], batch_size=64, num_workers=32)
        query_dataset = inference_d.make_iterative_query_inference_dataset(self.config, wandb_run.config)
        query_embeddings, query_labels = embeddings_builder.build_embeddings(self.config, wandb_run, query_dataset)
        corpus_dataset = inference_d.make_iterative_corpus_inference_dataset(self.config, wandb_run.config)
        if wandb_run.config['dry']:
            corpus_dataset = Subset(corpus_dataset, indices=range(10000))
        corpus_embeddings, corpus_paths = embeddings_builder.build_embeddings(self.config, wandb_run, corpus_dataset)

        metric = utils.get_metric(wandb_run.config)
        retriever = FaissRetriever(embeddings_size=corpus_embeddings.shape[1], metric=metric)
        retriever.add_to_index(corpus_embeddings, labels=corpus_paths)
        distances, matched_paths = retriever.query(query_embeddings, k=self.iterative_config['images_by_iterations'])

        curated_builder = CuratedBuilder(
            self.iterative_config,
            self.path_iterative_data,
            score_mode='distance'
        )
        curated_builder(
            query_labels, matched_paths, distances,
            f'{wandb_run.name}-{wandb_run.id}'
        )

        fit_dict['iterative_data'] = f'{wandb_run.name}-{wandb_run.id}.json'


class CuratedBuilder:
    def __init__(self,
                 iterative_config: dict,
                 folder: str,
                 curated_dataset: list = None,
                 score_mode: str = 'distance'):
        self.curated_dataset = [] if curated_dataset is None else curated_dataset
        self.path = folder
        self.iterative_config = iterative_config
        self.score_mode = score_mode

    def build(self,
              query_labels: np.ndarray,
              matched_paths: np.ndarray,
              scores: np.ndarray):
        query_labels = np.asarray(query_labels)
        matched_paths = np.asarray(matched_paths)
        scores = np.asarray(scores)

        # validate shapes of inputs
        if len(query_labels.shape) != 1:
            raise ValueError(f'Expected query_labels to be 1-dimensional array, got {query_labels.shape} instead')

        if matched_paths.shape != (query_labels.shape[0], self.iterative_config["images_by_iterations"]):
            raise ValueError(
                f'Expected matched_paths to have shape {(query_labels.shape[0], self.iterative_config["images_by_iterations"])}, got {matched_paths.shape} instead')

        if scores.shape != (query_labels.shape[0], self.iterative_config['images_by_iterations']):
            raise ValueError(
                f'Expected {self.score_mode}_scores to have shape {(query_labels.shape[0], self.iterative_config["images_by_iterations"])}, got {scores.shape} instead')

        for i, x in enumerate(query_labels):
            image_paths = matched_paths[i]
            image_scores = scores[i]
            curated_images = [image_dict['image_path'] for image_dict in self.curated_dataset]
            for image_path, image_score in zip(image_paths, image_scores):
                if self.iterative_config['duplicate_threshold'] < image_score < self.iterative_config[
                    'curated_threshold']:
                    if image_path not in curated_images:
                        self.curated_dataset.append({
                            'label': x,
                            'image_path': image_path,
                            self.score_mode: float(image_score)
                        })
                        curated_images.append(image_path)

        return self

    def to_json(self, json_name: str = 'curated_dataset') -> None:
        path = os.path.join(self.path, f'{json_name}.json')
        with open(path, 'w+') as f:
            json.dump(self.curated_dataset, f)

    def __call__(self,
                 query_labels,
                 matched_paths,
                 scores,
                 json_name: str = 'curated_dataset') -> None:

        self.build(query_labels, matched_paths, scores)
        self.to_json(json_name)


if __name__ == '__main__':
    main()
