import warnings

warnings.filterwarnings('ignore')

import warnings

import os
import torch
import wandb

import src.models.vit.lightning as vit_l
from src import utils
from src.models import utils as mutils

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')


def main():
    config = utils.get_config()
    wandb_config = utils.init_wandb('vit.yml')
    trainer = mutils.get_trainer(config, wandb_config)
    lightning = get_lightning(config, wandb_config)
    trainer.fit(model=lightning)
    wandb.finish()


def get_lightning(config, wandb_config, checkpoint=None):
    model = vit_l.get_model(wandb_config)

    kargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model
    }

    if checkpoint is None:
        lightning = vit_l.RefConLightning(**kargs)
    else:
        path_checkpoint = os.path.join(config['path']['models'], checkpoint)
        lightning = vit_l.RefConLightning.load_from_checkpoint(path_checkpoint, **kargs)

    return lightning


if __name__ == '__main__':
    main()
