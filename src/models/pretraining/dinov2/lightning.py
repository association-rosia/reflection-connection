import torch

import pytorch_lightning as pl
from transformers import ViTModel
from src.models.losses import DINOLoss, iBOTLoss, KoLeoLoss
from torch.utils.data import DataLoader
import src.data.datasets.dinov2 as dinov2_d
from src import utils
from src.models.modules import RefConHead


class RefConLightning(pl.LightningModule):
    def __init__(self, config: dict, wandb_config: dict):
        super().__init__()
        self.config = config
        self.wandb_config = wandb_config

        self.student_vit = ViTModel.from_pretrained(self.wandb_config['model_id'], use_mask_token=True)
        self.teacher_vit = ViTModel.from_pretrained(self.wandb_config['model_id'], use_mask_token=True)

        self.student_head = RefConHead(768, self.wandb_config['num_prototypes'])
        self.teacher_head = RefConHead(768, self.wandb_config['num_prototypes'])

        self.dino_loss = DINOLoss(self.wandb_config)
        self.ibot_loss = iBOTLoss(self.wandb_config)
        self.koleo_loss = KoLeoLoss()

        self.freeze_teacher_params()

    def dino_forward(self, batch, set):
        student_outputs = self.student_vit(pixel_values=batch['dino_student_inputs'])
        teacher_outputs = self.teacher_vit(pixel_values=batch['dino_teacher_inputs'])
        dino_student_ps = self.student_head(student_outputs.last_hidden_state[:, 0])

        with torch.no_grad():
            dino_teacher_logits = self.teacher_head(teacher_outputs.last_hidden_state[:, 0])
            dino_teacher_ps = self.dino_loss.softmax_center(dino_teacher_logits)

            if set == 'train':
                self.dino_loss.update_center(dino_teacher_logits)

        return dino_student_ps, dino_teacher_ps  # DINO prototype scores

    def ibot_forward(self, batch, set):
        bool_masked_pos = batch['ibot_bool_masked_pos']
        student_outputs = self.student_vit(pixel_values=batch['ibot_inputs'], bool_masked_pos=bool_masked_pos)
        teacher_outputs = self.teacher_vit(pixel_values=batch['ibot_inputs'])
        ibot_student_ps = self.student_head(student_outputs.last_hidden_state)

        with torch.no_grad():
            ibot_teacher_logits = self.teacher_head(teacher_outputs.last_hidden_state)
            ibot_teacher_ps = self.ibot_loss.softmax_center(ibot_teacher_logits)

            if set == 'train':
                self.ibot_loss.update_center(ibot_teacher_logits)

        return ibot_student_ps, ibot_teacher_ps  # iBOT prototype scores

    def training_step(self, batch, batch_idx):
        loss = 0

        dino_student_ps, dino_teacher_ps = self.dino_forward(batch, set='train')
        dino_loss = self.dino_loss(dino_student_ps, dino_teacher_ps)
        loss += dino_loss

        ibot_student_ps, ibot_teacher_ps = self.ibot_forward(batch, set='train')
        ibot_loss = self.ibot_loss(ibot_student_ps, ibot_teacher_ps, batch['ibot_bool_masked_pos'])
        loss += ibot_loss

        koleo_student_cls = ibot_student_ps[:, 0, :]
        koleo_loss = self.koleo_loss(koleo_student_cls)
        loss += 0.2 * koleo_loss

        self.update_teacher()

        self.log_dict({
            'train/dino_loss': dino_loss,
            'train/ibot_loss': ibot_loss,
            'train/koleo_loss': koleo_loss,
            'train/loss': loss
        })

        return loss

    def validation_step(self, batch, batch_idx):
        loss = 0

        dino_student_ps, dino_teacher_ps = self.dino_forward(batch, set='val')
        dino_loss = self.dino_loss(dino_student_ps, dino_teacher_ps)
        loss += dino_loss

        ibot_student_ps, ibot_teacher_ps = self.ibot_forward(batch, set='val')
        ibot_loss = self.ibot_loss(ibot_student_ps, ibot_teacher_ps, batch['ibot_bool_masked_pos'])
        loss += ibot_loss

        koleo_student_cls = ibot_student_ps[:, 0, :]
        koleo_loss = self.koleo_loss(koleo_student_cls)
        loss += 0.1 * koleo_loss

        self.log_dict({
            'val/dino_loss': dino_loss,
            'val/ibot_loss': ibot_loss,
            'val/koleo_loss': koleo_loss,
            'val/loss': loss
        })

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student_vit.parameters(), lr=self.wandb_config['lr'])
        return optimizer

    def freeze_teacher_params(self):
        for param in self.teacher_vit.parameters():
            param.requires_grad = False

        for param in self.teacher_head.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, teacher_momentum=0.994):
        for teacher_param, student_param in zip(self.teacher_vit.parameters(), self.student_vit.parameters()):
            teacher_param.data = teacher_momentum * teacher_param.data + (1 - teacher_momentum) * student_param.data

        for teacher_param, student_param in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            teacher_param.data = teacher_momentum * teacher_param.data + (1 - teacher_momentum) * student_param.data

    def train_dataloader(self):
        dataset = dinov2_d.make_petrain_dataset(self.config, self.wandb_config, set='train')

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=True,
        )

        return dataloader

    def val_dataloader(self):
        dataset = dinov2_d.make_petrain_dataset(self.config, self.wandb_config, set='val')

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=False,
        )

        return dataloader


def _debug():
    config = utils.get_config()
    wandb_config = utils.init_wandb('pretrain.yml')

    kwargs = {
        'config': config,
        'wandb_config': wandb_config
    }

    lightning = RefConLightning(**kwargs)

    return


if __name__ == '__main__':
    _debug()
