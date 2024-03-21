import warnings

warnings.filterwarnings('ignore')

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Any

from transformers import ViTMAEForPreTraining

import src.data.datasets.vitmae as vitmae_td
from src import utils

import wandb


class RefConLightning(pl.LightningModule):
    def __init__(
            self,
            config: dict,
            wandb_config: dict,
            model: ViTMAEForPreTraining,
            *args: Any,
            **kwargs: Any
    ):
        super(RefConLightning, self).__init__()
        self.config = config
        self.wandb_config = wandb_config
        self.model = model

    def forward(self, inputs):
        return self.model(pixel_values=inputs)

    def training_step(self, batch):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)

        if batch_idx == 0:
            self.log_image(batch, outputs)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.wandb_config['lr'])
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True),
            'monitor': 'val/loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        if self.wandb_config['type'] == 'pretraining':
            dataset = vitmae_td.make_pretrain_dataset(self.config, self.wandb_config, set='train')
        elif self.wandb_config['type'] == 'fine-tuning':
            dataset, _ = vitmae_td.make_fine_tuned_dataset(self.config, self.wandb_config)
        else:
            raise ValueError(f'Unknown training type: {self.config["type"]}')

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self):
        if self.wandb_config['type'] == 'pretraining':
            dataset = vitmae_td.make_pretrain_dataset(self.config, self.wandb_config, set='val')
        elif self.wandb_config['type'] == 'fine-tuning':
            _, dataset = vitmae_td.make_fine_tuned_dataset(self.config, self.wandb_config)
        else:
            raise ValueError(f'Unknown training type: {self.config["type"]}')

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=False,
        )

        return dataloader


def get_model(wandb_config) -> ViTMAEForPreTraining:
    model = ViTMAEForPreTraining.from_pretrained(pretrained_model_name_or_path=wandb_config['model_id'])

    return model


def _debug():
    config = utils.get_config()
    wandb_config = utils.init_wandb('training/vitmae.yml')
    model = get_model(wandb_config)

    kwargs = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model
    }

    lightning = RefConLightning(**kwargs)

    return


if __name__ == '__main__':
    _debug()
