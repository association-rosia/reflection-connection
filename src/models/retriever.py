from typing import overload
from typing_extensions import Self
from numpy._typing import ArrayLike
import torch
import numpy as np
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
        faiss.normalize_L2(embeddings)
        
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
        if labels is not None:
            assert embeddings.shape[0] == len(labels)
            self.labels = np.concatenate([self.labels, labels], axis=0)

        embeddings = self._preprocess_embeddings(embeddings)
        index = self._make_index()

        index.add(embeddings)
        if self.index.ntotal == 0:
            self.index = index
        else:
            self.index.merge_from(index)
        
        return self


def _debug():
    from src import utils
    from src.models.inference import InferenceModel, EmbeddingsBuilder
    import os
    
    config = utils.get_config()
    wandb_run = utils.get_run('96t0rkbl')
    model = InferenceModel.load_from_wandb_run(config, wandb_run, 'cpu')
    embeddings_builder = EmbeddingsBuilder(device=0, return_names=True)
    query_folder_path = os.path.join(config['path']['data'], 'raw', 'test', 'query')
    corpus_folder_path = os.path.join(config['path']['data'], 'raw', 'test', 'image_corpus')
    corpus_embeddings, corpus_names = embeddings_builder.build_embeddings(model=model, folder_path=corpus_folder_path, return_names=True)
    query_embeddings, query_names = embeddings_builder.build_embeddings(model=model, folder_path=query_folder_path, return_names=True)

    del model, embeddings_builder
    torch.cuda.empty_cache()
    
    retriever = FaissRetriever(embeddings_size=corpus_embeddings.shape[1])
    retriever.add_to_index(corpus_embeddings, corpus_names)
    cosine_distance, corpus_names_retrived = retriever.query(query_embeddings, k=3)
    
    return 0


if __name__ == '__main__':
    _debug()