import os
import torch
import torchvision.transforms.functional as tvF
import wandb
import yaml


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    return device


def load_config(yml_file):
    root = os.path.join('config', yml_file)
    notebooks = os.path.join(os.pardir, root)
    path = root if os.path.exists(root) else notebooks

    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config():
    return load_config('config.yml')


def init_wandb(yml_file):
    wandb_config = load_config(yml_file)
    config = get_config()

    wandb.init(
        entity=config['wandb']['entity'],
        project=config['wandb']['project'],
        config=wandb_config
    )

    return wandb.config


def get_run(run_id: str):
    run = None

    if run_id:
        project_config = get_config()

        api = wandb.Api()
        run = wandb.apis.public.Run(
            client=api.client,
            entity=project_config['wandb']['entity'],
            project=project_config['wandb']['project'],
            run_id=run_id,
        )

    return run


class RunDemo:
    def __init__(self, config_file, id, name) -> None:
        self.config = load_config(config_file)
        self.name = name
        self.id = id


def resize_tensor_2d(tensor, size, interpolation=tvF.InterpolationMode.BILINEAR):
    resized_tensor = tvF.resize(tensor.unsqueeze(0), size=size, interpolation=interpolation).squeeze(0)

    return resized_tensor

