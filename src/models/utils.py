import os

import pytorch_lightning as pl
import wandb

import src.models.fine_tuning.clip.lightning as ft_clip
import src.models.fine_tuning.dinov2.lightning as ft_dinov2
import src.models.fine_tuning.vitmae.lightning as ft_vit_mae
import src.models.fine_tuning.vit.torchvision.lightning as ft_vit_torchvision
import src.models.fine_tuning.vit.transformers.lightning as ft_vit_transformers

import src.models.pretraining.dinov2.lightning as pt_dinov2
import src.models.pretraining.vitmae.lightning as pt_vitmae


def get_lightning_library(model_id, training='fine_tuning'):
    if training == 'fine_tuning':
        if 'clip' in model_id:
            return ft_clip
        elif 'dinov2' in model_id:
            return ft_dinov2
        elif 'vit-mae' in model_id:
            return ft_vit_mae
        elif 'ViT' in model_id:
            return ft_vit_torchvision
        elif 'vit' in model_id:
            return ft_vit_transformers
        else:
            raise NotImplementedError()
    elif training == 'pretraining':
        if 'vit-mae' in model_id:
            return pt_vitmae
        elif 'vit' in model_id:
            return pt_dinov2
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def get_lightning(config, wandb_config, training='fine_tuning', checkpoint=None):
    lightning_module = get_lightning_library(wandb_config['model_id'], training=training)
    model = lightning_module.get_model(wandb_config=wandb_config, config=config)

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


def get_trainer(config):
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
        patience=50,
        verbose=True,
        mode='min'
    )

    if wandb.config.dry:
        trainer = pl.Trainer(
            max_epochs=2,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            devices=wandb.config.devices,
            precision='16-mixed',
            limit_train_batches=3,
            limit_val_batches=3
        )
    elif len(wandb.config.devices) > 1:
        trainer = pl.Trainer(
            devices=wandb.config.devices,
            max_epochs=wandb.config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback, early_stopping_callback],
            precision='16-mixed',
            strategy='ddp_find_unused_parameters_true',
            val_check_interval=wandb.config.val_check_interval
        )
    else:
        trainer = pl.Trainer(
            devices=wandb.config.devices,
            max_epochs=wandb.config.max_epochs,
            logger=pl.loggers.WandbLogger(),
            callbacks=[checkpoint_callback, early_stopping_callback],
            precision='16-mixed',
            val_check_interval=wandb.config.val_check_interval
        )

    return trainer
