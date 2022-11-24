
# Solution for 11th place of Yandex Cup 2022

The task of the track was to predict tracks of the same artist as the anchor track.

My solution is based on ensemble of [TransformerEncoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html) and [Conformer](https://pytorch.org/audio/stable/generated/torchaudio.models.Conformer.html#torchaudio.models.Conformer).

I used metric learning and learned similarity of the artist with [MultiSimilarityLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multisimilarityloss), [CircleLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss) and [TripletMarginLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss)

Sum of different losses helps to find global optimum more smoothly

Also, I used Curriculum Learning with increasing length of sequence from 10 to 81 steps. It helps to generalize model.

Batches were divided with respect to artist, so different batches couldn't contain same artist.

As for augmentations, I used random zeroing and random permutation of channels and timesteps.

On the first model evaluations averaging model outputs on rolling window of timesteps(for example 81 timesteps could be divided into 15 windows with 10 steps and 5 hop) helped

Also, I added inference reranking. First, we get top-100 tracks according to anchor track. Then, we take tracks that closer than 0.85 of cosine similarity. Then average them with original track and get top-100 again without tracks that we already took.


```text
$ python3 train_transformer.py
Training: epoch: 81/100, loss: 2.9784581661224365, max_ndcg: 0.4750643992621211: 100%|██████████| 14/14 [01:32<00:00,  6.57s/it] 
Training: epoch: 12/100, loss: 2.966454267501831, max_ndcg: 0.47831202115003973: 100%|██████████| 14/14 [01:30<00:00,  6.48s/it] 
Training: epoch: 10/100, loss: 2.906022310256958, max_ndcg: 0.48753636409671103: 100%|██████████| 17/17 [01:42<00:00,  6.04s/it] 
Training: epoch: 7/100, loss: 2.849691390991211, max_ndcg: 0.4885703528191847: 100%|██████████| 24/24 [01:58<00:00,  4.92s/it]  
Training: epoch: 8/100, loss: 2.796380043029785, max_ndcg: 0.49302302476400534: 100%|██████████| 34/34 [02:03<00:00,  3.63s/it] 
Training: epoch: 7/100, loss: 2.7387564182281494, max_ndcg: 0.4937553819406174: 100%|██████████| 42/42 [02:11<00:00,  3.14s/it]
Training: epoch: 6/100, loss: 2.6662023067474365, max_ndcg: 0.49523512491856597: 100%|██████████| 84/84 [02:36<00:00,  1.87s/it]
Training: epoch: 8/100, loss: 2.671266555786133, max_ndcg: 0.4952613059116226: 100%|██████████| 84/84 [02:32<00:00,  1.82s/it] 
Training: epoch: 34/100, loss: 2.7556495666503906, max_ndcg: 0.5006257613883551: 100%|██████████| 84/84 [03:08<00:00,  2.24s/it]
```

```text
$ python3 train_conformer.py
Training: epoch: 7/100, loss: 3.0670297145843506, max_ndcg: 0.2715291981582186: 100%|██████████| 17/17 [01:25<00:00,  5.02s/it]
Training: epoch: 33/100, loss: 2.9149909019470215, max_ndcg: 0.4208720136192471: 100%|██████████| 42/42 [01:38<00:00,  2.35s/it] 
Training: epoch: 6/100, loss: 2.8750646114349365, max_ndcg: 0.44926664024477414: 100%|██████████| 42/42 [01:51<00:00,  2.65s/it]
Training: epoch: 7/100, loss: 2.818598985671997, max_ndcg: 0.46274198297569213: 100%|██████████| 56/56 [02:02<00:00,  2.19s/it] 
Training: epoch: 7/100, loss: 2.7469844818115234, max_ndcg: 0.4680077147865726: 100%|██████████| 84/84 [02:25<00:00,  1.74s/it] 
Training: epoch: 8/100, loss: 2.6955888271331787, max_ndcg: 0.46927703067015525: 100%|██████████| 111/111 [03:04<00:00,  1.66s/it]
Training: epoch: 7/100, loss: 2.7038466930389404, max_ndcg: 0.4696899116316955: 100%|██████████| 111/111 [03:20<00:00,  1.80s/it]
Training: epoch: 32/100, loss: 2.7776288986206055, max_ndcg: 0.47590464302588764: 100%|██████████| 111/111 [03:39<00:00,  1.98s/it]
```

```text
$ python3 test.py
transformer - Validation NDCG: 0.5006257135701774
transformer - Validation NDCG with rerank: 0.5121140426735709
confomer - Validation NDCG: 0.4759069667172343
confomer - Validation NDCG with rerank: 0.4880699387550571
ensemble - Validation NDCG: 0.5119186970150843
ensemble - Validation NDCG with rerank: 0.5229597764821007
```