from torch.optim import AdamW
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import CLIPVisionModelWithProjection

import src.data.make_dataset as md
from src import utils


class RefConLightning(pl.LightningModule):
    def __init__(self, args):
        super(RefConLightning, self).__init__()
        self.config = args['config']
        self.wandb_config = args['wandb_config']
        self.model = args['model']
        self.train_indices = args['train_indices']
        self.val_indices = args['val_indices']
        self.dataset = args['dataset']

        self.criterion = TripletMarginLoss(swap=True)
    
    def forward(self, anchors, positives, negatives):
        anchors_embed = self.model(pixel_values=anchors)['image_embeds']
        positives_embed = self.model(pixel_values=positives)['image_embeds']
        negatives_embed = self.model(pixel_values=negatives)['image_embeds']
        loss = self.criterion(anchors_embed, positives_embed, negatives_embed)
        
        return loss
        
    def training_step(self, batch):
        loss = self.forward(*batch)
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch):
        loss = self.forward(*batch)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(params=self.model.parameters(), lr=self.wandb_config['lr'])

        return optimizer

    def train_dataloader(self):
        dataset = md.RefConDataset(
            config=self.config,
            wandb_config=self.wandb_config,
            dataset=self.dataset,
            indices=self.train_indices,
            train=True
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self):
        dataset = md.RefConDataset(
            config=self.config,
            wandb_config=self.wandb_config,
            dataset=self.dataset,
            indices=self.val_indices,
            train=False
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=False,
        )

        return dataloader


def get_model(wandb_config):
    model = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path=wandb_config['model_id'],
        ignore_mismatched_sizes=True
    )

    return model


if __name__ == '__main__':
    config = utils.get_config()
    wandb_config = utils.init_wandb('clip.yml')
    model = get_model(wandb_config)
    base_dataset = md.get_base_dataset(config)
    train_indices, val_indices = md.get_train_val_indices(wandb_config, base_dataset)

    args = {
        'config': config,
        'wandb_config': wandb_config,
        'model': model,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'dataset': base_dataset
    }

    lightning = RefConLightning(args)

    pass
