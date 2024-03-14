import os

import pytorch_lightning as pl
import wandb
import src.models.vit.lightning as vit_l
import src.models.clip.lightning as clip
import src.models.dinov2.lightning as dinov2


def get_lightning_library(wandb_config):
    if 'clip' in wandb_config['model_id']:
        return clip
    elif 'dinov2' in wandb_config['model_id']:
        return dinov2
    elif 'ViT' in wandb_config['model_id']:
        return vit_l
    else:
        raise NotImplementedError()


def get_lightning(config, wandb_config, checkpoint=None):
    model = vit_l.get_model(wandb_config)

    kwargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model
    }

    if checkpoint is None:
        lightning = vit_l.RefConLightning(**kwargs)
    else:
        path_checkpoint = os.path.join(config['path']['models'], checkpoint)
        lightning = vit_l.RefConLightning.load_from_checkpoint(path_checkpoint, **kwargs)

    return lightning


def get_trainer(config, devices):
    os.makedirs(config['path']['models'], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        dirpath=config['path']['models'],
        filename=f'{wandb.run.name}-{wandb.run.id}',
        auto_insert_metric_name=False,
        verbose=True
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val/loss',
        patience=100,
        verbose=True,
        mode='min'
    )

    if wandb.config.dry:
        trainer = pl.Trainer(
            max_epochs=3,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            devices=devices,
            precision='16-mixed',
            limit_train_batches=5,
            limit_val_batches=5
        )
    else:
        trainer = pl.Trainer(
            devices=devices,
            max_epochs=wandb.config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback, early_stopping_callback],
            precision='16-mixed'
        )

    return trainer
