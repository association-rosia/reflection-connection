import warnings

warnings.filterwarnings('ignore')

import warnings

import os
import torch
import wandb

import src.models.pretraining.dinov2.lightning as dinov2_l
from src import utils
from src.models import utils as mutils

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')


def main():
    config = utils.get_config()
    wandb_config = utils.init_wandb('pretraining/dinov2.yml')
    trainer = mutils.get_trainer(config, wandb_config)
    lightning = get_lightning(config, wandb_config)
    trainer.fit(model=lightning)
    wandb.finish()


def get_lightning(config, wandb_config, checkpoint=None):
    kwargs = {
        'config': config,
        'wandb_config': wandb_config
    }

    if checkpoint is None:
        lightning = dinov2_l.RefConLightning(**kwargs)
    else:
        path_checkpoint = os.path.join(config['path']['models'], checkpoint)
        lightning = dinov2_l.RefConLightning.load_from_checkpoint(path_checkpoint, **kwargs)

    return lightning


if __name__ == '__main__':
    main()
