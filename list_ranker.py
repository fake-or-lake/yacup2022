import numpy as np
import torch
import torch.nn.functional as F


def get_ranked_list(embeds, top_size):
    keys = np.array(list(embeds.keys()))
    embeddings = torch.cat([embeds[i].unsqueeze(0).cuda() for i in keys], dim=0)
    ranked_list = {}
    for i, (track_id, track_embed) in enumerate(embeds.items()):
        sim = F.cosine_similarity(embeddings[i:i + 1], embeddings, dim=1)
        sorted_top = sim.topk(top_size + 1)[1].cpu().numpy()
        ranked_list[track_id] = [i for i in keys[sorted_top] if i != track_id]
    return ranked_list


def get_ranked_list_rerank_threshold(embeds, top_size, threshold=0.85):
    keys = np.array(list(embeds.keys()))
    embeddings = torch.cat([embeds[i].unsqueeze(0).cuda() for i in keys], dim=0)
    ranked_list = {}
    for i, (track_id, track_embed) in enumerate(embeds.items()):
        sim = F.cosine_similarity(embeddings[i:i + 1], embeddings, dim=1)
        top_101 = sim.topk(top_size + 1)
        ranked_keys = top_101[1][top_101[0] > threshold].cpu().numpy()
        threshold_list = [i for i in keys[ranked_keys] if i != track_id]

        average_embedding = torch.mean(embeddings[ranked_keys], dim=0, keepdim=True)
        sim = F.cosine_similarity(average_embedding, embeddings, dim=1)
        top_101 = sim.topk(top_size + 1)[1].cpu().numpy()
        ranked_list[track_id] = threshold_list
        ranked_list[track_id] += [i for i in keys[top_101] if i not in threshold_list and i != track_id][
                                 :top_size - len(threshold_list)]
    return ranked_list
