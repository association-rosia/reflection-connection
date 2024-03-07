import torch
from typing import Self, overload
import numpy as np
from numpy._typing import ArrayLike
import faiss


class FaissRetriever:
    def __init__(self, embeddings_size: int, metric: str = 'cosine', use_gpu: bool = False) -> None:
        self.metric = metric
        self.embeddings_size = embeddings_size
        self.use_gpu = use_gpu
        self.index = self._make_index()
        self.labels = np.array([])
        
    def _get_faiss_metric(self):
        if self.metric == 'l2':
            return faiss.METRIC_L2
        if self.metric == 'cosine':
            return faiss.METRIC_INNER_PRODUCT
        
    def _make_index(self):
        faiss_metric = self._get_faiss_metric()
        index = faiss.index_factory(self.embeddings_size, "Flat", faiss_metric)
        if self.use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)
        
        return index

    @staticmethod
    def _preprocess_embeddings(embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy(force=True)
        if len(embeddings.shape) == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        embeddings = embeddings.astype(np.float32)
        embeddings = faiss.normalize_L2(embeddings)
        
        return embeddings
    
    def query(self, embeddings: np.ndarray | torch.Tensor, k: int = 4):
        embeddings = self._preprocess_embeddings(embeddings)
        similarity, indices = self.index.search(embeddings, k)
        
        if len(self.labels) == self.index.ntotal:
            return 1 - similarity, self.labels[indices]
        else:
            return 1 - similarity, indices

    @overload
    def add_to_index(self, embeddings: np.ndarray | torch.Tensor) -> Self: ...
    @overload
    def add_to_index(self, embeddings: np.ndarray | torch.Tensor, labels: ArrayLike) -> Self: ...
    def add_to_index(self, embeddings: np.ndarray | torch.Tensor, labels: ArrayLike) -> Self:
        embeddings = self._preprocess_embeddings(embeddings)
        index = self._make_index()
        if labels is not None:
            assert embeddings.shape[0] == len(labels)
            self.labels = np.concatenate([self.labels, labels], axis=0)
        
        index.add(embeddings)
        if self.index.ntotal == 0:
            self.index = index
        else:
            self.index.merge_from(index)
        
        return self


# class SearchBruteForce:
#     def __init__(self, corpus_set: ImageSet) -> None:
#         embeddings = corpus_set.get_embeddings()
#         self.corpus_embedings = np.stack([embedding[0].numpy(force=True) for embedding in embeddings])
#         self.corpus_labels = np.stack([embedding[1] for embedding in embeddings])
        
#     def query(self, query_set: ImageSet, metric: str = 'l2', k: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         embeddings = query_set.get_embeddings()
#         query_embedings = np.stack([embedding[0].numpy(force=True) for embedding in embeddings])
#         query_labels = np.stack([embedding[1] for embedding in embeddings])
        
#         matrix_distances = pairwise_distances(query_embedings, self.corpus_embedings, metric=metric, n_jobs=-1)
#         indices = []
#         distances = []
#         for query_distances in matrix_distances:
#             indices.append(np.argsort(query_distances)[:k])
#             distances.append(query_distances[indices[-1]])
        
#         distances = np.stack(distances)
#         indices = np.stack(indices)
        
#         return query_labels, distances, self.corpus_embedings[indices], self.corpus_labels[indices]
    
#     def __call__(self, *args, **kwargs) -> Any:
#         return self.query(*args, **kwargs)


# if __name__ == '__main__':
#     config = utils.get_config()
#     wandb_run = utils.get_run('2khs9u4f')

#     query_set = ImageSet(config, wandb_run, query=True, cuda_idx=0)
#     corpus_set = ImageSet(config, wandb_run, query=False, cuda_idx=0)
#     query_set.build_embeddings()
#     corpus_set.build_embeddings()
#     # sbf = SearchBruteForce(corpus_set)
#     # sbf.query(query_set)