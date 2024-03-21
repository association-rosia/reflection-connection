import os
from glob import glob
from src import utils
from sklearn.model_selection import train_test_split


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


def get_class_path(dir_path):
    list_class_name = []
    list_img_path = []
    for class_name in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_path, img_name)
                    list_class_name.append(class_name)
                    list_img_path.append(img_path)

    return list_class_name, list_img_path


def get_image_folder(config):
    path = os.path.join(config['path']['data'], 'raw', 'train')
    path = utils.get_notebooks_path(path)

    return path


def get_train_val_split(wandb_config, list_class_name, list_img_path):
    train_class_name, val_class_name, train_path_img, val_path_img = train_test_split(
        list_class_name, list_img_path,
        train_size=0.8,
        random_state=wandb_config['random_state'],
        stratify=list_class_name
    )

    return train_class_name, val_class_name, train_path_img, val_path_img
