import torch

from train import train_model
from model import YaCupModelTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YaCupModelTransformer(n_features=512,
                              projection_dim=128).to(device)

curriculum_params = [
    {"batch_size": 1200, "crop_size": 10, "num_epochs": 100, "lr": 3e-4, "augment_tracks": False},
    {"batch_size": 1200, "crop_size": 10, "num_epochs": 100, "lr": 1e-4, "augment_tracks": False},
    {"batch_size": 1000, "crop_size": 20, "num_epochs": 100, "lr": 1e-4, "augment_tracks": False},
    {"batch_size": 700, "crop_size": 30, "num_epochs": 100, "lr": 1e-4, "augment_tracks": False},
    {"batch_size": 500, "crop_size": 40, "num_epochs": 100, "lr": 1e-5, "augment_tracks": False},
    {"batch_size": 400, "crop_size": 60, "num_epochs": 100, "lr": 1e-5, "augment_tracks": False},
    {"batch_size": 200, "crop_size": 81, "num_epochs": 100, "lr": 1e-5, "augment_tracks": False},
    {"batch_size": 200, "crop_size": 81, "num_epochs": 100, "lr": 1e-6, "augment_tracks": False},
    {"batch_size": 200, "crop_size": 81, "num_epochs": 100, "lr": 1e-6, "augment_tracks": True},
]

train_model(model, curriculum_params, device, "transformer")
