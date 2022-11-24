import numpy as np
import torch

from model import YaCupAveragingModel


def position_discounter(position):
    return 1.0 / np.log2(position + 1)


def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg


def compute_dcg(query_trackid, ranked_list, track2artist_map, top_size):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        assert result_trackid != query_trackid
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg


def eval_submission(submission, gt_meta_info, top_size=100):
    track2artist_map = gt_meta_info.set_index('trackid')['artistid'].to_dict()
    artist2tracks_map = gt_meta_info.groupby('artistid').agg(list)['trackid'].to_dict()
    ndcg_list = []
    for query_trackid in submission.keys():
        ranked_list = submission[query_trackid]
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count - 1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map, top_size=top_size)
        try:
            ndcg_list.append(dcg / ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)


def inference(model, loader, averaging=False, crop_size=None, step=None):
    model.eval()
    embeds = dict()
    if averaging:
        inference_model = YaCupAveragingModel(model, crop_size, step)
    else:
        inference_model = model
    for tracks_ids, (tracks_features, lengths) in loader:
        with torch.no_grad():
            tracks_embeds = inference_model(tracks_features, lengths)
        for track_id, track_embed in zip(tracks_ids, tracks_embeds):
            embeds[track_id] = track_embed
    return embeds


def inference_and_rank(module, val_loader, ranker, top_size=100):
    embeds = inference(module, val_loader)
    ranked_list = ranker(embeds, top_size)
    return ranked_list


def get_ndcg(module, val_loader, valset_meta, ranker, top_size=100):
    ranked_list = inference_and_rank(module, val_loader, ranker, top_size)
    return eval_submission(ranked_list, valset_meta)
