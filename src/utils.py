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


def load_config(yml_file: str) -> dict:
    root = os.path.join('config', yml_file)
    path = get_notebooks_path(root)

    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config() -> dict:
    return load_config('config.yml')


def init_wandb(yml_file: str) -> dict:
    config = get_config()
    wandb_dir = get_notebooks_path(config['path']['logs']['root'])
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = os.path.abspath(wandb_dir)
    
    wandb_config = load_config(yml_file)

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
    def __init__(self, config_file: str, id: str, name: str) -> None:
        self.config = load_config(config_file)
        self.name = name
        self.id = id
