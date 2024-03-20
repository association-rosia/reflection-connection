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
        outputs = self.model(pixel_values=inputs)
        loss = outputs.loss

        return loss

    def training_step(self, batch):
        loss = self.forward(batch)
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch):
        loss = self.forward(batch)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.wandb_config['lr'])
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True),
            'monitor': 'val/loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = vitmae_td.make_pretrain_dataset(self.config, self.wandb_config, set='train')

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self):
        dataset = vitmae_td.make_pretrain_dataset(self.config, self.wandb_config, set='val')

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
    wandb_config = utils.init_wandb('pretraining/vitmae.yml')
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
