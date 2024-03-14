import torch
from torch import nn
import pytorch_lightning as pl
from transformers import ViTModel
from src.models.losses import DINOLoss
from torch.utils.data import DataLoader
import src.data.datasets.pretrain_dataset as td
from src import utils


class RefConLightning(pl.LightningModule):
    def __init__(self, config: dict, wandb_config: dict):
        super().__init__()
        self.config = config
        self.wandb_config = wandb_config

        self.student_vit = ViTModel.from_pretrained(self.wandb_config['model_id'])
        self.teacher_vit = ViTModel.from_pretrained(self.wandb_config['model_id'])

        self.student_dino_head = RefConHead(768, self.wandb_config['num_prototypes'])
        self.teacher_dino_head = RefConHead(768, self.wandb_config['num_prototypes'])
        self.dino_loss = DINOLoss()

        self.student_ibot_head = RefConHead(768, self.wandb_config['num_prototypes'])
        self.teacher_ibot_head = RefConHead(768, self.wandb_config['num_prototypes'])

        self.freeze_teacher_params()

    # def forward(self, pixel_values):
    #     student_outputs = self.student_vit(pixel_values=pixel_values)
    #     student_outputs = self.student_head(student_outputs.last_hidden_state[:, 0])
    #
    #     return student_outputs

    def dino_forward(self, student_outputs, teacher_outputs):
        dino_student_logits = self.student_dino_head(student_outputs.last_hidden_state[:, 0])
        dino_student_ps = torch.softmax(dino_student_logits, dim=-1)

        with torch.no_grad():
            dino_teacher_logits = self.teacher_dino_head(teacher_outputs.last_hidden_state[:, 0])
            dino_teacher_ps = self.sinkhorn_knopp(dino_teacher_logits)

        return dino_student_ps, dino_teacher_ps  # DINO prototype scores

    def training_step(self, batch, batch_idx):
        loss = 0

        student_outputs = self.student_vit(pixel_values=batch['dino_student_inputs'])
        teacher_outputs = self.teacher_vit(pixel_values=batch['dino_teacher_inputs'])

        dino_student_ps, dino_teacher_ps = self.dino_forward(student_outputs, teacher_outputs)
        loss += self.dino_loss(dino_student_ps, dino_teacher_ps)

        self.update_teacher()

        self.log('train/loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student_vit.parameters(), lr=self.wandb_config['lr'])
        return optimizer

    def freeze_teacher_params(self):
        for param in self.teacher_vit.parameters():
            param.requires_grad = False

        for param in self.student_dino_head.parameters():
            param.requires_grad = False

        for param in self.teacher_ibot_head.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def sinkhorn_knopp(tensor, teacher_temp=0.07, iterations=3):
        Q = torch.exp(tensor / teacher_temp).t()
        Q /= Q.sum()

        u = torch.zeros(Q.shape[0]).to(tensor.device)
        r = torch.ones(Q.shape[0]).to(tensor.device) / Q.shape[0]
        c = torch.ones(Q.shape[1]).to(tensor.device) / Q.shape[1]

        for _ in range(iterations):
            u = Q.sum(dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(0)

        Q /= Q.sum(dim=0, keepdim=True)

        return Q.t()

    @torch.no_grad()
    def update_teacher(self, teacher_momentum=0.992):
        for teacher_param, student_param in zip(self.teacher_vit.parameters(), self.student_vit.parameters()):
            teacher_param.data = teacher_momentum * teacher_param.data + (1 - teacher_momentum) * student_param.data

        for teacher_param, student_param in zip(self.teacher_dino_head.parameters(),
                                                self.student_dino_head.parameters()):
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
