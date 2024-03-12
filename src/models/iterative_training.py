import os
from src import utils
import src.models.utils as mutils
from src.models.retriever import FaissRetriever
from src.models.inference import InferenceModel, EmbeddingsBuilder
from src.submissions import make_submissions
import wandb

def main():
    config = utils.get_config()
    iterative_config = utils.load_config('iterative.yml')
    iterative_data = None
    for _ in range(iterative_config['iterations']):
        utils.init_wandb(iterative_config['model_config'])
        train_model(config, iterative_data)
        create_next_iterative_dataset(config, iterative_config['images_by_iterations'])
        wandb.finish()


def train_model(config, iterative_data):
    wandb.config.update({'iterative_data': iterative_data})
    trainer = mutils.get_trainer(config)
    lightning = mutils.get_lightning(config, wandb.config)
    trainer.fit(model=lightning)


def create_next_iterative_dataset(config, images_by_iterations):
    model = InferenceModel.load_from_wandb_run(config, wandb.run, 'cpu')
    embeddings_builder = EmbeddingsBuilder(device=0, return_names=True)
    query_folder_path = os.path.join(config['path']['data'], 'raw', 'train')
    corpus_folder_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'train')
    corpus_embeddings, corpus_names = embeddings_builder.build_embeddings(model=model, folder_path=corpus_folder_path, return_names=True)
    query_embeddings, query_names = embeddings_builder.build_embeddings(model=model, folder_path=query_folder_path, return_names=True)
    
    metric = make_submissions.get_metric(wandb.config)
    retriever = FaissRetriever(embeddings_size=corpus_embeddings.shape[1], metric=metric)
    retriever.add_to_index(corpus_embeddings, labels=corpus_names)
    distances, matched_labels = retriever.query(query_embeddings, k=images_by_iterations)
    confidence_scores = make_submissions.dist_to_conf(distances)
    
    # Create submission file
    path_iterative_data = os.path.join(config['path']['data'], 'processed', 'train')
    result_builder = make_submissions.ResultBuilder(path_iterative_data, k=images_by_iterations)
    result_builder(
        query_names,
        matched_labels,
        confidence_scores,
        f'{wandb.run.name}-{wandb.run.id}'
    )