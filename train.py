import os
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import MultiSimilarityLoss, MultipleLosses, CircleLoss, TripletMarginLoss
from tqdm import tqdm

from data import TestLoader, FeaturesLoader, TrainLoaderFactory
from list_ranker import get_ranked_list
from metrics import get_ndcg
from utils import seed_all, train_val_split


def train(module, train_generator, val_loader, valset_meta, criterion, base_dir, batch_size, crop_size, num_epochs, lr,
          augment_tracks,
          name,
          top_size=100):
    checkpoint_path = base_dir / f"{name}_{crop_size}_{lr}_best.pt"
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.001)
    train_loader = train_generator.get_loader(batch_size, crop_size, augment_tracks)
    max_ndcg = 0
    tolerance = 5
    bar = tqdm(total=len(train_loader))
    for epoch in range(num_epochs):
        gbl_loss = 0
        bar.reset()
        for i, batch in enumerate(train_loader):
            module.train()

            (x, lengths), labels = batch
            z = module(x, lengths)
            loss = criterion(z, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            gbl_loss += loss.detach()
            bar.set_description(
                f"Training: epoch: {epoch + 1}/{num_epochs}, loss: {gbl_loss / (i + 1)}, max_ndcg: {max_ndcg}")
            bar.update(1)

        with torch.no_grad():
            module.eval()
            val_ndcg = get_ndcg(module, val_loader, valset_meta, get_ranked_list, top_size)
            if val_ndcg > max_ndcg:
                tolerance = 5
                max_ndcg = val_ndcg
                torch.save(module.state_dict(), checkpoint_path)
            else:
                tolerance -= 1
        if tolerance == 0:
            break
    return checkpoint_path


def train_model(model, curriculum_params, device, name):
    seed_all(42)

    base_dir = Path("solution_results")
    os.makedirs(base_dir, exist_ok=True)
    train_set = Path('train_features/train_features')
    train_meta = Path('train_meta.tsv')

    train_meta_info = pd.read_csv(train_meta, sep='\t')
    train_meta_info, validation_meta_info = train_val_split(train_meta_info, val_size=0.1)

    loss = MultipleLosses(
        losses=[
            MultiSimilarityLoss(alpha=2, beta=50, base=1),
            CircleLoss(m=0.4, gamma=80),
            TripletMarginLoss(margin=0.05,
                              swap=True,
                              smooth_loss=True,
                              triplets_per_anchor=10,
                              distance=CosineSimilarity())
        ],
        weights=[
            1,
            0.025,
            1,
        ])

    train_generator = TrainLoaderFactory(FeaturesLoader(train_set, train_meta_info, device))
    val_loader = TestLoader(FeaturesLoader(train_set, validation_meta_info, device), batch_size=200, crop_size=81)

    train_loop = partial(train, module=model,
                         train_generator=train_generator,
                         val_loader=val_loader,
                         valset_meta=validation_meta_info,
                         criterion=loss,
                         base_dir=base_dir,
                         name=name)

    for params in curriculum_params:
        checkpoint_path = train_loop(**params)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
