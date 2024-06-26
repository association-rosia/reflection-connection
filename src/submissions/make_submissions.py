import json
import os

import numpy as np

import src.data.datasets.inference as inference_d
from src import utils
from src.models.inference import EmbeddingsBuilder
from src.models.retriever import FaissRetriever


def main():
    config = utils.get_config()
    wandb_run = utils.get_run('yqbxbjmd')
    embeddings_builder = EmbeddingsBuilder(devices=[0])

    corpus_dataset = inference_d.make_submission_corpus_inference_dataset(config, wandb_run.config)
    corpus_embeddings, corpus_names = embeddings_builder.build_embeddings(config, wandb_run, dataset=corpus_dataset)
    query_dataset = inference_d.make_submission_query_inference_dataset(config, wandb_run.config)
    query_embeddings, query_names = embeddings_builder.build_embeddings(config, wandb_run, dataset=query_dataset)

    metric = utils.get_metric(wandb_run.config)
    retriever = FaissRetriever(embeddings_size=corpus_embeddings.shape[1], metric=metric)
    retriever.add_to_index(corpus_embeddings, labels=corpus_names)
    distances, matched_labels = retriever.query(query_embeddings, k=3)
    confidence_scores = dist_to_conf(distances)

    # Create submission file
    result_builder = ResultBuilder(config['path']['submissions'], k=3)
    result_builder(
        query_names,
        matched_labels,
        confidence_scores,
        f'{wandb_run.name}-{wandb_run.id}'
    )


def dist_to_conf(distances: np.ndarray):
    max_dist = distances.max(axis=1).reshape(-1, 1)
    min_dist = distances.min(axis=1).reshape(-1, 1)
    distances_normalized = (distances - min_dist) / max_dist
    confidence_from_dist = 1 - distances_normalized

    return confidence_from_dist


class ResultBuilder:
    def __init__(self, path, k: int = 3, score_mode: str = 'confidence'):
        self.results = dict()
        self.path = utils.get_notebooks_path(path)
        self.k = k
        self.score_mode = score_mode

    def build(self,
              query_image_labels: np.ndarray,
              matched_labels: np.ndarray,
              scores: np.ndarray):
        query_image_labels = np.asarray(query_image_labels)
        matched_labels = np.asarray(matched_labels)
        scores = np.asarray(scores)

        # validate shapes of inputs
        if len(query_image_labels.shape) != 1:
            raise ValueError(
                f'Expected query_image_labels to be 1-dimensional array, got {query_image_labels.shape} instead')

        if matched_labels.shape != (query_image_labels.shape[0], self.k):
            raise ValueError(
                f'Expected matched_labels to have shape {(query_image_labels.shape[0], self.k)}, got {matched_labels.shape} instead')

        if scores.shape != (query_image_labels.shape[0], self.k):
            raise ValueError(
                f'Expected {self.score_mode}_scores to have shape {(query_image_labels.shape[0], self.k)}, got {scores.shape} instead')

        for i, x in enumerate(query_image_labels):
            labels = matched_labels[i]
            confidence = scores[i]

            result_x = [{'label': labels[j], self.score_mode: float(confidence[j])} for j in range(0, self.k)]

            self.results.update({x: result_x})

        return self

    def to_json(self, json_name: str = 'results') -> None:

        path = os.path.join(self.path, f'{json_name}.json')
        with open(path, 'w+') as f:
            json.dump(self.results, f)

    def __call__(self,
                 query_image_labels,
                 matched_labels,
                 scores,
                 json_name: str = 'results') -> None:

        self.build(query_image_labels, matched_labels, scores)
        self.to_json(json_name)


if __name__ == '__main__':
    main()
