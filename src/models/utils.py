import os

import pytorch_lightning as pl
import wandb
import src.models.vit.torchvision.lightning as vit_torchvision
import src.models.vit.transformers.lightning as vit_transformers
import src.models.clip.lightning as clip
import src.models.dinov2.lightning as dinov2


def get_lightning_library(model_id):
    if 'clip' in model_id:
        return clip
    elif 'dinov2' in model_id:
        return dinov2
    elif 'ViT' in model_id:
        return vit_torchvision
    elif 'vit' in model_id:
        return vit_transformers
    else:
        raise NotImplementedError()


def get_lightning(config, wandb_config, checkpoint=None):
    lightning_module = get_lightning_library(wandb_config)
    model = lightning_module.get_model(wandb_config)

    kwargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model
    }

    if checkpoint is None:
        lightning = lightning_module.RefConLightning(**kwargs)
    else:
        path_checkpoint = os.path.join(config['path']['models'], checkpoint)
        lightning = lightning_module.RefConLightning.load_from_checkpoint(path_checkpoint, **kwargs)

    return lightning


def get_trainer(config, wandb_config):
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
            devices=1,
            precision='16-mixed',
            limit_train_batches=5,
            limit_val_batches=5
        )
    elif len(wandb.config.devices) > 1:
        trainer = pl.Trainer(
            devices=wandb.config.devices,
            max_epochs=wandb.config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            precision='16-mixed',
            strategy='ddp_find_unused_parameters_true',
            val_check_interval=wandb_config['val_check_interval']
        )
    else:
        trainer = pl.Trainer(
            devices=wandb.config.devices,
            max_epochs=wandb.config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback, early_stopping_callback],
            precision='16-mixed',
            val_check_interval=wandb_config['val_check_interval']
        )

    return trainer
