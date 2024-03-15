import torch
from torch import nn
import pytorch_lightning as pl
from transformers import ViTModel
from src.models.losses import DINOiBOTLoss
from torch.utils.data import DataLoader
import src.data.datasets.pretrain_dataset as td
from src import utils


class RefConLightning(pl.LightningModule):
    def __init__(self, config: dict, wandb_config: dict):
        super().__init__()
        self.config = config
        self.wandb_config = wandb_config

        self.student_vit = ViTModel.from_pretrained(self.wandb_config['model_id'], use_mask_token=True)
        self.teacher_vit = ViTModel.from_pretrained(self.wandb_config['model_id'])

        self.student_head = RefConHead(768, self.wandb_config['num_prototypes'])
        self.teacher_head = RefConHead(768, self.wandb_config['num_prototypes'])

        self.dino_loss = DINOiBOTLoss()
        self.ibot_loss = DINOiBOTLoss()

        self.freeze_teacher_params()

    def dino_forward(self, batch):
        student_outputs = self.student_vit(pixel_values=batch['dino_student_inputs'])
        teacher_outputs = self.teacher_vit(pixel_values=batch['dino_teacher_inputs'])

        dino_student_logits = self.student_head(student_outputs.last_hidden_state[:, 0])
        dino_student_ps = torch.softmax(dino_student_logits, dim=-1)

        with torch.no_grad():
            dino_teacher_logits = self.teacher_head(teacher_outputs.last_hidden_state[:, 0])
            dino_teacher_ps = self.sinkhorn_knopp(dino_teacher_logits)

        return dino_student_ps, dino_teacher_ps  # DINO prototype scores

    def ibot_forward(self, batch):
        bool_masked_pos = batch['ibot_bool_masked_pos']
        student_outputs = self.student_vit(pixel_values=batch['ibot_inputs'], bool_masked_pos=bool_masked_pos)
        teacher_outputs = self.teacher_vit(pixel_values=batch['ibot_inputs'])

        ibot_student_logits = self.student_head(student_outputs.last_hidden_state)
        ibot_student_ps = torch.softmax(ibot_student_logits, dim=-1)
        print(ibot_student_ps.shape)

        with torch.no_grad():
            dino_teacher_logits = self.teacher_head(teacher_outputs.last_hidden_state)
            print(dino_teacher_logits.shape)
            ibot_teacher_ps = self.sinkhorn_knopp(dino_teacher_logits)

        return ibot_student_ps, ibot_teacher_ps  # iBOT prototype scores

    def training_step(self, batch, batch_idx):
        loss = 0

        dino_student_ps, dino_teacher_ps = self.dino_forward(batch)
        dino_loss = self.dino_loss(dino_student_ps, dino_teacher_ps)
        loss += dino_loss

        ibot_student_ps, ibot_teacher_ps = self.ibot_forward(batch)
        ibot_loss = self.ibot_loss(ibot_student_ps, ibot_teacher_ps)
        loss += ibot_loss

        self.update_teacher()

        self.log_dict({
            'train/dino_loss': dino_loss,
            'train/ibot_loss': ibot_loss,
            # 'train/koleo_loss': koleo_loss,
            'train/loss': loss
        })

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student_vit.parameters(), lr=self.wandb_config['lr'])
        return optimizer

    def freeze_teacher_params(self):
        for param in self.teacher_vit.parameters():
            param.requires_grad = False

        for param in self.student_head.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def sinkhorn_knopp(self, tensor, teacher_temp=0.07, iterations=3):
        tensor = torch.exp(tensor / teacher_temp)
        batch_size, num_rows, num_cols = tensor.shape

        # Original normalization adapted for 3D tensor
        Q = tensor.view(batch_size, num_rows, num_cols)
        Q /= Q.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        r = torch.ones((batch_size, num_rows), device=tensor.device) / num_rows
        c = torch.ones((batch_size, num_cols), device=tensor.device) / num_cols

        for _ in range(iterations):
            u = Q.sum(dim=2)
            Q *= (r / u).unsqueeze(-1)
            Q *= (c / Q.sum(dim=1, keepdim=True)).unsqueeze(-2)

        Q /= Q.sum(dim=2, keepdim=True).sum(dim=1, keepdim=True)

        # If you need to transpose the last two dimensions back to their original order
        # Adjust this if your model's subsequent operations require a different dimension order
        Q = Q.permute(0, 2, 1)  # This switches back the last two dimensions

        return Q

    @torch.no_grad()
    def update_teacher(self, teacher_momentum=0.994):
        for teacher_param, student_param in zip(self.teacher_vit.parameters(), self.student_vit.parameters()):
            teacher_param.data = teacher_momentum * teacher_param.data + (1 - teacher_momentum) * student_param.data

        for teacher_param, student_param in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            teacher_param.data = teacher_momentum * teacher_param.data + (1 - teacher_momentum) * student_param.data

    def train_dataloader(self):
        dataset = td.make_petrain_dataset(self.config, self.wandb_config)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.wandb_config['batch_size'],
            num_workers=self.config['archi']['num_workers'],
            drop_last=True,
            shuffle=True,
        )

        return dataloader


class RefConHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.head(x)


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
