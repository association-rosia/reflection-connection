import json
import numpy as np
from src.submissions import search
from src import utils
import os


def main():
    config = utils.get_config()
    wandb_run = utils.get_run('sxa0zzzr')
    query_set = search.ImageSet(config, wandb_run, query=True, cuda_idx=0)
    corpus_set = search.ImageSet(config, wandb_run, query=False, cuda_idx=0)
    query_set.build_embeddings()
    corpus_set.build_embeddings()
    sbf = search.SearchBruteForce(corpus_set)
    metric = get_metric(wandb_run.config)
    query_image_labels, distances, _, matched_labels = sbf.query(query_set, metric=metric)
    confidence_scores = dist_to_conf(distances)
    result_builder = ResultBuilder(config)
    result_builder = result_builder(
        query_image_labels,
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

    distances_normalized = (distances - min_dist)/max_dist
    confidence_from_dist = 1 - distances_normalized
    
    return confidence_from_dist


class ResultBuilder:
    def __init__(self, config):
        self.results = dict()
        path = config['path']['submissions']['root']
        self.path = utils.get_notebooks_path(path)
        
    def build(self, 
              query_image_labels: np.ndarray, 
              matched_labels: np.ndarray,   
              confidence_scores: np.ndarray):
            
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
    
            result_x = [{'label': labels[j], 'confidence': confidence[j]} for j in range(0,3)]
    
            self.results.update({x: result_x})
        
        return self
    
    def to_json(self, json_name: str = 'results') -> None:
        
        path = os.path.join(self.path, f'{json_name}.json')
        with open(path, 'w+') as f:
            json.dump(self.results, f)
    
    def __call__(self,
              query_image_labels: np.ndarray, 
              matched_labels: np.ndarray,   
              confidence_scores: np.ndarray,
              json_name: str = 'results') -> None:
        
        self.build(query_image_labels, matched_labels, confidence_scores)
        self.to_json(json_name)


if __name__ == '__main__':
    main()