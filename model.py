import torch
import torchaudio
from torch import nn


class YaCupModelTransformer(nn.Module):
    def __init__(self, n_features, projection_dim):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_features, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

    def forward(self, x, lengths=None):
        x = torch.permute(x, [0, 2, 1])
        x = self.encoder(x).mean(axis=1)
        x = self.projector(x)
        return x


class YaCupModelConformer(nn.Module):
    def __init__(self, n_features, projection_dim):
        super().__init__()
        self.encoder = torchaudio.models.Conformer(
            input_dim=n_features,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=31,
        )
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
        )

    def forward(self, x, lengths=None):
        x = torch.permute(x, [0, 2, 1])
        x = self.encoder(x, lengths)[0].mean(axis=1)
        x = self.projector(x)
        return x


class YaCupModelEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def forward(self, x, lengths=None):
        outputs = [model(x, lengths) for model in self.models]
        return torch.cat(outputs, dim=1)

class YaCupAveragingModel(nn.Module):
    def __init__(self, model, crop_size, step):
        super().__init__()
        self.model = model
        self.crop_size = crop_size
        self.step = step

    def forward(self, x, lengths=None):
        current_shape = x.shape[-1]
        track_embeds = []
        while current_shape >= self.crop_size:
            slice = x[:, :, current_shape - self.crop_size:current_shape]
            slice_lengths = torch.tensor([slice.shape[-1]] * slice.shape[0]).to(x.device)
            track_embeds.append(self.model(slice, slice_lengths).unsqueeze(0))
            current_shape -= self.step
        return torch.mean(torch.cat(track_embeds, dim=0), dim=0)