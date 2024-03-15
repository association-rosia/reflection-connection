import warnings

warnings.filterwarnings('ignore')

import os
import warnings

import torch
import wandb

import src.models.dinov2.lightning as dinov2_l
from src import utils
from src.models import utils as mutils

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')


def main():
    config = utils.get_config()
    wandb_config = utils.init_wandb('dinov2.yml')
    trainer = mutils.get_trainer(config, devices=[0])
    lightning = get_lightning(config, wandb_config)
    trainer.fit(model=lightning)
    wandb.finish()


def get_lightning(config, wandb_config, checkpoint=None):
    model = dinov2_l.get_model(wandb_config)

    kargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model
    }

    if checkpoint is None:
        lightning = dinov2_l.RefConLightning(**kargs)
    else:
        path_checkpoint = os.path.join(config['path']['models']['root'], checkpoint)
        lightning = dinov2_l.RefConLightning.load_from_checkpoint(path_checkpoint, **kargs)

    return lightning


if __name__ == '__main__':
    main()
