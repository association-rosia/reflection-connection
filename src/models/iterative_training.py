import os
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
    iterative_data = None
    for _ in range(iterative_config['iterations']):
        utils.init_wandb(iterative_config['model_config'])
        wandb.config.update({
            'iterative_data': iterative_data, 
            'curated_treshold': iterative_config['curated_treshold'],
            'duplicate_treshold': iterative_config['duplicate_treshold'],
        })
        train_model(config)
        iterative_data = create_next_iterative_dataset(config, iterative_config)
        wandb.finish()


def train_model(config):
    trainer = mutils.get_trainer(config)
    lightning = mutils.get_lightning(config, wandb.config)
    trainer.fit(model=lightning)


def create_next_iterative_dataset(config, iterative_config, iterative_data):
    model = InferenceModel.load_from_wandb_run(config, wandb.run, 'cpu')
    embeddings_builder = EmbeddingsBuilder(device=0, return_names=True)
    
    curated_dataset = utils.load_curated_dataset(iterative_data)
    query_folder_path = get_query_images_paths(config, curated_dataset)
    corpus_folder_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'train')
    # passer en mode liste de path pour les iteration suivante
    # il faut récupérer les paths depuis la racine
    # ajouter les paths des images curated 
    corpus_embeddings, corpus_names = embeddings_builder.build_embeddings(model=model, folder_path=corpus_folder_path, return_names=True)
    query_embeddings, query_names = embeddings_builder.build_embeddings(model=model, folder_path=query_folder_path, return_names=True)
    query_classes = get_query_classes(iterative_data, query_names)
    
    metric = utils.get_metric(wandb.config)
    retriever = FaissRetriever(embeddings_size=corpus_embeddings.shape[1], metric=metric)
    retriever.add_to_index(corpus_embeddings, labels=corpus_names)
    distances, matched_labels = retriever.query(query_embeddings, k=iterative_config['images_by_iterations'])
    
    # Create submission file
    path_iterative_data = os.path.join(config['path']['data'], 'processed', 'train')
    result_builder = CuratedBuilder(
        path_iterative_data,
        k=iterative_config['images_by_iterations'],
        score_mode='distance'
    )
    result_builder(
        query_classes,
        matched_labels,
        distances,
        f'{wandb.run.name}-{wandb.run.id}'
    )
    
    return f'{wandb.run.name}-{wandb.run.id}.json'


def get_query_images_paths(config, curated_dataset: dict):
    template_path = os.path.join(config['path']['data'], 'raw', 'train', '**', '*.png')
    query_images_paths = glob(template_path, recursive=True)
    if curated_dataset is not None:
        filtered_paths = [
            image_dict['image_path'] for image_dict in image_dicts 
            if duplicate_threshold < image_dict['distance'] < curated_threshold
        ]
    return query_images_paths


def get_class_for_image(image_name, class_folders):
    for folder in class_folders:
        if image_name in os.listdir(folder):
            return os.path.basename(folder)
    return None

def get_classes_for_images(image_names, parent_folder):
    class_folders = [os.path.join(parent_folder, folder) for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]
    classes = [get_class_for_image(image_name, class_folders) for image_name in image_names]
    return classes

# lire le fichier iterative_data en json et récupérer la class correspondante à chaque image
# faire un set entre tous les fichier de la query et tous les fichiers du corpus 
# corpus = set(corpus) - set(query)
def get_query_classes(iterative_data):
    pass    

# faire un json avec chaques classes et à l'interrieur
# liste de dict[classe: dict[image_path: str, distance: str]]
class CuratedBuilder:
    def __init__(self, path, k: int = 3, score_mode: str = 'distance'):
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
            raise ValueError(f'Expected query_image_labels to be 1-dimensional array, got {query_image_labels.shape} instead')
        
        if matched_labels.shape != (query_image_labels.shape[0], self.k):
            raise ValueError(f'Expected matched_labels to have shape {(query_image_labels.shape[0], self.k)}, got {matched_labels.shape} instead')
        
        if scores.shape != (query_image_labels.shape[0], self.k):
            raise ValueError(f'Expected {self.score_mode}_scores to have shape {(query_image_labels.shape[0], self.k)}, got {scores.shape} instead')
            
        for i, x in enumerate(query_image_labels):
            labels = matched_labels[i]
            confidence = scores[i]
    
            result_x = [{'label': labels[j], self.score_mode: float(confidence[j])} for j in range(0,3)]
    
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
