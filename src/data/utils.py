import os
import json
from glob import glob
from src import utils
from sklearn.model_selection import train_test_split

def load_augmented_dataset(wandb_config):
    if wandb_config.get('iterative_data', None) is None:
        return []
    config = utils.get_config()
    path = os.path.join(config['path']['data'], 'processed', 'train', wandb_config['iterative_data'])
    path = utils.get_notebooks_path(path)
    with open(path, 'r') as f:
        return json.load(f)

def get_images_path(config, set):
    train_path = os.path.join(config['path']['data'], 'raw', 'train')
    train_glob = os.path.join(train_path, '**/*.png')
    test_path = os.path.join(config['path']['data'], 'raw', 'test')
    test_glob = os.path.join(test_path, '**/*.png')
    images_path = glob(train_glob, recursive=True) + glob(test_glob, recursive=True)

    if set == 'train':
        pretrain_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
        pretrain_glob = os.path.join(pretrain_path, '**/*.png')
        images_path += glob(pretrain_glob, recursive=True)

    return images_path


def get_paths_labels(folder_path):
    labels = []
    paths = []
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_path, img_name)
                    labels.append(class_name)
                    paths.append(img_path)
    
    return paths, labels


def get_image_folder(config):
    path = os.path.join(config['path']['data'], 'raw', 'train')
    path = utils.get_notebooks_path(path)

    return path

def get_curated_class_path(config):
    image_folder = get_image_folder(config)
    curated_image_paths, curated_labels = get_paths_labels(image_folder)
    
    return curated_image_paths, curated_labels


def get_train_val_split(wandb_config, image_paths, labels):
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        train_size=0.8,
        random_state=wandb_config['random_state'],
        stratify=labels
    )

    return train_image_paths, val_image_paths, train_labels, val_labels