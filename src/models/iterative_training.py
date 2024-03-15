import os
import json
from glob import glob
import numpy as np
from src import utils
import src.models.utils as mutils
from src.models.retriever import FaissRetriever
from src.models.inference import InferenceModel, EmbeddingsBuilder
import wandb


def main():
    config = utils.get_config()
    iterative_config = utils.load_config('iterative.yml')
    uncurated_folder = os.path.join(config['path']['data'], 'processed', 'pretrain')
    curated_folder = os.path.join(config['path']['data'], 'raw', 'train')
    
    iterative_trainer = IterativeTrainer(
        config,
        iterative_config,
        curated_folder,
        uncurated_folder,
        device=2
    )
    
    iterative_trainer.fit()
    
class IterativeTrainer:
    def __init__(self,
                 config,
                 iterative_config,
                 curated_folder,
                 uncurated_folder,
                 device) -> None:
        self.config = config
        self.iterative_config = iterative_config
        self.curated_folder = curated_folder
        self.uncurated_folder = uncurated_folder
        self.device = device
        self.path_iterative_data = os.path.join(self.config['path']['data'], 'processed', 'train')
    
    def fit(self):
        iterative_data = None
        for _ in range(self.iterative_config['iterations']):
            utils.init_wandb(self.iterative_config['model_config'])
            wandb.config.update({
                'iterative_data': iterative_data, 
                'curated_treshold': self.iterative_config['curated_treshold'],
                'duplicate_treshold': self.iterative_config['duplicate_treshold'],
            })
            self._train_model()
            iterative_data = self._create_next_iterative_dataset()
            wandb.finish()

    def _train_model(self):
        trainer = mutils.get_trainer(self.config, devices=[self.device])
        lightning = mutils.get_lightning(self.config, wandb.config)
        trainer.fit(model=lightning)

    def _create_next_iterative_dataset(self):
        model = InferenceModel.load_from_wandb_run(self.config, wandb.run, 'cpu')
        embeddings_builder = EmbeddingsBuilder(device=self.device, return_names=True)
        query_paths, query_labels = self._get_query_paths_labels()
        corpus_paths = self._get_corpus_paths(query_paths)
        corpus_embeddings = embeddings_builder.build_embeddings(model=model, list_paths=corpus_paths, return_names=False)
        query_embeddings = embeddings_builder.build_embeddings(model=model, list_paths=query_paths, return_names=False)
        
        metric = utils.get_metric(wandb.config)
        retriever = FaissRetriever(embeddings_size=corpus_embeddings.shape[1], metric=metric)
        retriever.add_to_index(corpus_embeddings, labels=corpus_paths)
        distances, matched_paths = retriever.query(query_embeddings, k=self.iterative_config['images_by_iterations'])
        
        # Create submission file
        
        curated_builder = CuratedBuilder(
            self.path_iterative_data,
            k=self.iterative_config['images_by_iterations'],
            score_mode='distance'
        )
        curated_builder(
            query_labels,
            matched_paths,
            distances,
            f'{wandb.run.name}-{wandb.run.id}'
        )
        
        return f'{wandb.run.name}-{wandb.run.id}.json'

    def _get_query_paths_labels(self):
        augmented_dataset = utils.load_augmented_dataset(wandb.config)
        query_paths, query_labels = utils.get_paths_labels(self.curated_folder)
        query_paths.extend([image_dict['image_path'] for image_dict in augmented_dataset])
        query_labels.extend([image_dict['label'] for image_dict in augmented_dataset])

        return query_paths, query_labels

    def _get_corpus_paths(self, query_paths):
        template_path = os.path.join(self.uncurated_folder, '*.png')
        corpus_paths = glob(template_path)
        
        return set(corpus_paths) - set(query_paths)


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
            raise ValueError(f'Expected matched_paths to have shape {(query_labels.shape[0], self.iterative_config["images_by_iterations"])}, got {matched_paths.shape} instead')
        
        if scores.shape != (query_labels.shape[0], self.iterative_config['images_by_iterations']):
            raise ValueError(f'Expected {self.score_mode}_scores to have shape {(query_labels.shape[0], self.iterative_config["images_by_iterations"])}, got {scores.shape} instead')
            
        for i, x in enumerate(query_labels):
            image_paths = matched_paths[i]
            image_scores = scores[i]
            curated_images = [image_dict['image_path'] for image_dict in self.curated_dataset]
            for image_path, image_score in zip(image_paths, image_scores):
                if self.iterative_config['duplicate_threshold'] < image_score < self.iterative_config['curated_threshold']:
                    if image_path not in curated_images:
                        self.curated_dataset.append({
                            'label': x,
                            'image_path': image_path,
                            self.score_mode: image_score
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