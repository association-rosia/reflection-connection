import warnings

warnings.filterwarnings('ignore')

import warnings

import os
import torch
import wandb

import src.models.pretrain.lightning as pretrain_l
from src import utils
from src.models import utils as mutils

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')


def main():
    config = utils.get_config()
    wandb_config = utils.init_wandb('pretrain.yml')
    trainer = mutils.get_trainer(config)
    lightning = get_lightning(config, wandb_config)
    trainer.fit(model=lightning)
    wandb.finish()


def get_lightning(config, wandb_config, checkpoint=None):
    kargs = {
        'config': config,
        'wandb_config': wandb_config
    }

    if checkpoint is None:
        lightning = pretrain_l.RefConLightning(**kargs)
    else:
        path_checkpoint = os.path.join(config['path']['models'], checkpoint)
        lightning = pretrain_l.RefConLightning.load_from_checkpoint(path_checkpoint, **kargs)

    return lightning


if __name__ == '__main__':
    main()
