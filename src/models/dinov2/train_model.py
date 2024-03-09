import warnings

warnings.filterwarnings('ignore')

import os
import warnings

import pytorch_lightning as pl
import torch
import wandb

import src.models.dinov2.lightning as dinov2_l
from src import utils

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('medium')


def main():
    config = utils.get_config()
    wandb_config = utils.init_wandb('dinov2.yml')
    trainer = get_trainer(config)
    lightning = get_lightning(config, wandb_config)
    trainer.fit(model=lightning)
    wandb.finish()


def get_trainer(config):
    os.makedirs(config['path']['models']['root'], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        dirpath=config['path']['models']['root'],
        filename=f'{wandb.run.name}-{wandb.run.id}',
        auto_insert_metric_name=False,
        verbose=True
    )

    if wandb.config.dry:
        trainer = pl.Trainer(
            max_epochs=3,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            devices=1,
            precision='16-mixed',
            limit_train_batches=5,
            limit_val_batches=5
        )
    else:
        trainer = pl.Trainer(
            devices=1,
            max_epochs=wandb.config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            precision='16-mixed'
        )

    return trainer


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
