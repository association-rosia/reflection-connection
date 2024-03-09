import json
import os

import numpy as np


class ResultBuilder:
    def __init__(self):
        self.results = dict()

    def build(self,
              query_image_labels: np.ndarray,
              matched_labels: np.ndarray,
              confidence_scores: np.ndarray):

        # validate shapes of inputs
        if len(query_image_labels.shape) != 1:
            raise ValueError(
                f'Expected query_image_labels to be 1-dimensional array, got {query_image_labels.shape} instead')

        if matched_labels.shape != (query_image_labels.shape[0], 3):
            raise ValueError(
                f'Expected matched_labels to have shape {(query_image_labels.shape[0], 3)}, got {matched_labels.shape} instead')

        if confidence_scores.shape != (query_image_labels.shape[0], 3):
            raise ValueError(
                f'Expected confidence_scores to have shape {(query_image_labels.shape[0], 3)}, got {confidence_scores.shape} instead')

        for i, x in enumerate(query_image_labels):
            labels = matched_labels[i]
            confidence = confidence_scores[i]

            result_x = [{'label': labels[j], 'confidence': confidence[j]} for j in range(0, 3)]

            self.results.update({x: result_x})

        return self

    def to_json(self, path: str = os.curdir) -> None:

        path = f'{path}/results.json'
        with open(path, 'w+') as f:
            json.dump(self.results, f)

    def __call__(self,
                 query_image_labels: np.ndarray,
                 matched_labels: np.ndarray,
                 confidence_scores: np.ndarray,
                 path: str = os.curdir) -> None:

        self.build(query_image_labels, matched_labels, confidence_scores)
        self.to_json(path)
