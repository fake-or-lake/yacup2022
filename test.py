import os
from pathlib import Path

import pandas as pd
import torch

from data import TestLoader, FeaturesLoader
from list_ranker import get_ranked_list, get_ranked_list_rerank_threshold
from metrics import get_ndcg, inference_and_rank
from model import YaCupModelTransformer, YaCupModelConformer, YaCupModelEnsemble
from utils import seed_all, train_val_split, save_submission


def validate_model(model, val_loader, validation_meta_info, name):
    val_ndcg = get_ndcg(model, val_loader, validation_meta_info, get_ranked_list, top_size=100)
    print(f"{name} - Validation NDCG:", val_ndcg)
    rerank_dcg = get_ndcg(model, val_loader, validation_meta_info, get_ranked_list_rerank_threshold, top_size=100)
    print(f"{name} - Validation NDCG with rerank:", rerank_dcg)


if __name__ == "__main__":
    seed_all(42)
    base_dir = Path("solution_results")
    os.makedirs(base_dir, exist_ok=True)
    train_set = Path('train_features/train_features')
    train_meta = Path('train_meta.tsv')
    train_meta_info = pd.read_csv(train_meta, sep='\t')
    train_meta_info, validation_meta_info = train_val_split(train_meta_info, val_size=0.1)

    test_set = Path('test_features/test_features')
    test_set_meta = Path("test_meta.tsv")
    test_meta_info = pd.read_csv(test_set_meta, sep='\t')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    val_loader = TestLoader(FeaturesLoader(train_set, validation_meta_info, device), batch_size=2000, crop_size=81)
    test_loader = TestLoader(FeaturesLoader(test_set, test_meta_info, device), batch_size=2000, crop_size=81)

    transformer = YaCupModelTransformer(512, 128).to(device)
    weights_transformer = base_dir / "transformer_81_1e-06_best.pt"
    transformer.load_state_dict(torch.load(weights_transformer, map_location=device))
    validate_model(transformer, val_loader, validation_meta_info, "transformer")

    conformer = YaCupModelConformer(512, 128).to(device)
    weights_conformer = base_dir / "conformer_81_1e-06_best.pt"
    conformer.load_state_dict(torch.load(weights_conformer, map_location=device))
    validate_model(conformer, val_loader, validation_meta_info, "conformer")

    ensemble = YaCupModelEnsemble([transformer, conformer])
    validate_model(ensemble, val_loader, validation_meta_info, "ensemble")

    rerank_submission = inference_and_rank(ensemble, test_loader, get_ranked_list_rerank_threshold, top_size=100)
    save_submission(rerank_submission, base_dir / "submission.tsv")
