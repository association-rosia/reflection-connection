import os

import torch
import wandb
import wandb.apis.public as wandb_api
import yaml


def get_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    return device


def get_notebooks_path(path: str) -> str:
    notebooks = os.path.join(os.pardir, path)
    new_path = path if os.path.exists(path) else notebooks

    return new_path


def load_config(yml_file: str, sub_config: str = None) -> dict:
    root = os.path.join('configs', yml_file)
    path = get_notebooks_path(root)

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

        if sub_config is not None:
            config = config[sub_config]

    return config


def get_config() -> dict:
    return load_config('config.yml')


def init_wandb(yml_file: str, sub_config: str = None) -> dict:
    config = get_config()
    wandb_dir = get_notebooks_path(config['path']['logs'])
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ['WANDB_DIR'] = os.path.abspath(wandb_dir)

    wandb_config = load_config(yml_file, sub_config)

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        config=wandb_config
    )

    return wandb.config


def get_run(run_id: str) -> wandb_api.Run:
    run = None

    if run_id:
        project_config = get_config()

        api = wandb.Api()
        run = wandb_api.Run(
            client=api.client,
            entity=project_config['wandb']['entity'],
            project=project_config['wandb']['project'],
            run_id=run_id,
        )

    return run


class RunDemo:
    def __init__(self, config_file: str, id: str, name: str, sub_config: str = None) -> None:
        self.config = load_config(config_file, sub_config)
        self.name = name
        self.id = id


def get_metric(wandb_config):
    if wandb_config['criterion'] == 'TMWDL-Euclidean':
        return 'l2'
    elif wandb_config['criterion'] == 'TMWDL-Cosine':
        return 'cosine'


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
