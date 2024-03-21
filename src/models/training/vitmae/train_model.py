import warnings

warnings.filterwarnings('ignore')

import os
import warnings

import torch
import wandb

import src.models.training.vitmae.lightning as vitmae_l
from src import utils
from src.models import utils as mutils

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')


def main():
    config = utils.get_config()
    wandb_config = utils.init_wandb('training/vitmae.yml')
    trainer = mutils.get_trainer(config)
    lightning = get_lightning(config, wandb_config)
    trainer.fit(model=lightning)
    wandb.finish()


def get_lightning(config, wandb_config, checkpoint=None):
    model = vitmae_l.get_model(config, wandb_config)

    kwargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model
    }

    checkpoint = wandb_config.get('checkpoint', 'None')
    if checkpoint == 'None':
        lightning = vitmae_l.RefConLightning(**kwargs)
    else:
        path_checkpoint = os.path.join(config['path']['models'], checkpoint)
        lightning = vitmae_l.RefConLightning.load_from_checkpoint(path_checkpoint, **kwargs)

    return lightning


if __name__ == '__main__':
    main()
