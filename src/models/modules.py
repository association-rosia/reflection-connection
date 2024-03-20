from torch import nn


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


class RefConTorchvisionViT(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.head = RefConHead(1000, 1000)

    def forward(self, x):
        x = self.vit(x)
        x = self.head(x)

        return x


class RefConTransformersViT(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        self.head = RefConHead(768, 1000)

    def forward(self, x):
        x = self.vit(pixel_values=x)
        x = self.head(x.last_hidden_state[:, 0, :])

        return x
