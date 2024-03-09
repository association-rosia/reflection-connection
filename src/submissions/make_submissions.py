import json
import os

import numpy as np
from src import utils
from src.models.retriever import FaissRetriever
from src.models.inference import InferenceModel, EmbeddingsBuilder


def main():
    config = utils.get_config()
    wandb_run = utils.get_run('96t0rkbl')
    model = InferenceModel.load_from_wandb_run(config, wandb_run, 'cpu')
    embeddings_builder = EmbeddingsBuilder(device=0, return_names=True)
    query_folder_path = os.path.join(config['path']['data'], 'raw', 'test', 'query')
    corpus_folder_path = os.path.join(config['path']['data'], 'raw', 'test', 'image_corpus')
    corpus_embeddings, corpus_names = embeddings_builder.build_embeddings(model=model, folder_path=corpus_folder_path, return_names=True)
    query_embeddings, query_names = embeddings_builder.build_embeddings(model=model, folder_path=query_folder_path, return_names=True)
    
    metric = get_metric(wandb_run.config)
    retriever = FaissRetriever(embeddings_size=corpus_embeddings.shape[1], metric=metric)
    retriever.add_to_index(corpus_embeddings, labels=corpus_names)
    distances, matched_labels = retriever.query(query_embeddings, k=3)
    confidence_scores = dist_to_conf(distances)
    
    # Create submission file
    result_builder = ResultBuilder(config)
    result_builder(
        query_names,
        matched_labels,
        confidence_scores,
        f'{wandb_run.name}-{wandb_run.id}'
    )


def get_metric(wandb_config):
    if wandb_config['criterion'] == 'TMWDL-Euclidean':
        return 'l2'
    elif wandb_config['criterion'] == 'TMWDL-Cosine':
        return 'cosine'


def dist_to_conf(distances: np.ndarray):
    max_dist = distances.max(axis=1).reshape(-1, 1)
    min_dist = distances.min(axis=1).reshape(-1, 1)
    distances_normalized = (distances - min_dist) / max_dist
    confidence_from_dist = 1 - distances_normalized

    return confidence_from_dist


class ResultBuilder:
    def __init__(self, config):
        self.results = dict()
        path = config['path']['submissions']
        self.path = utils.get_notebooks_path(path)
        
    def build(self, 
              query_image_labels: np.ndarray, 
              matched_labels: np.ndarray,   
              confidence_scores: np.ndarray):
        query_image_labels = np.asarray(query_image_labels)
        matched_labels = np.asarray(matched_labels)
        confidence_scores = np.asarray(confidence_scores)
        
        # validate shapes of inputs
        if len(query_image_labels.shape) != 1:
            raise ValueError(f'Expected query_image_labels to be 1-dimensional array, got {query_image_labels.shape} instead')
        
        if matched_labels.shape != (query_image_labels.shape[0],3):
            raise ValueError(f'Expected matched_labels to have shape {(query_image_labels.shape[0], 3)}, got {matched_labels.shape} instead')
        
        if confidence_scores.shape != (query_image_labels.shape[0],3):
            raise ValueError(f'Expected confidence_scores to have shape {(query_image_labels.shape[0], 3)}, got {confidence_scores.shape} instead')
            
        for i, x in enumerate(query_image_labels):
            labels = matched_labels[i]
            confidence = confidence_scores[i]
    
            result_x = [{'label': labels[j], 'confidence': float(confidence[j])} for j in range(0,3)]
    
            self.results.update({x: result_x})
        
        return self
    
    def to_json(self, json_name: str = 'results') -> None:
        
        path = os.path.join(self.path, f'{json_name}.json')
        with open(path, 'w+') as f:
            json.dump(self.results, f)
    
    def __call__(self,
              query_image_labels, 
              matched_labels,   
              confidence_scores,
              json_name: str = 'results') -> None:
        
        self.build(query_image_labels, matched_labels, confidence_scores)
        self.to_json(json_name)


if __name__ == '__main__':
    main()
