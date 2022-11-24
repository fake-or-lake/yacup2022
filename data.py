import math
import os
import random

import numpy as np
import torch

from augmentations import augment_null_channels, augment_null_timestamps, augment_swap_timesteps, augment_swap_channels


def match_length(track, expected_length, mode="wrap"):
    original_length = track.shape[1]
    diff = np.abs(original_length - expected_length)
    if expected_length == original_length:
        return track, original_length
    elif expected_length < original_length:
        start = random.randint(0, diff)
        return track[:, start:start + expected_length], expected_length
    else:
        return np.pad(track, ((0, 0), (0, diff)), mode=mode), original_length


def augment(track, null_channels=True, null_timesteps=True, swap_channels=True, swap_timesteps=True):
    # (512, 81)
    if null_channels:
        track = augment_null_channels(track,
                                      max_count=random.randint(0, 10),
                                      probability=0.3)
    if null_timesteps:
        track = augment_null_timestamps(track,
                                        max_count=random.randint(0, 5),
                                        probability=0.3)
    if swap_channels:
        track = augment_swap_channels(track,
                                      max_count=random.randint(0, 10),
                                      probability=0.3)

    if swap_timesteps:
        track = augment_swap_timesteps(track,
                                       max_count=random.randint(0, 5),
                                       probability=0.3)
    return track


class FeaturesLoader:
    def __init__(self, features_dir_path, meta_info, device='cpu'):
        self.features_dir_path = features_dir_path
        self.meta_info = meta_info
        self.trackid2path = meta_info.set_index('trackid')['archive_features_path'].to_dict()
        self.device = device

    def preprocess_track_id(self, track_id, crop_size):
        track_features_file_path = self.trackid2path[track_id]
        track_features = np.load(os.path.join(self.features_dir_path, track_features_file_path))
        return match_length(track_features, crop_size)

    def _load_item(self, track_id, crop_size, augment_tracks):
        out, lengths = self.preprocess_track_id(track_id, crop_size)
        if augment_tracks:
            out = augment(out)
        return torch.tensor(out).unsqueeze(0), torch.tensor(lengths).unsqueeze(0)

    def load_batch(self, tracks_ids, crop_size, augment_tracks):
        batch_features = []
        batch_lengths = []
        for track_id in tracks_ids:
            features, length = self._load_item(track_id, crop_size=crop_size, augment_tracks=augment_tracks)
            batch_features.append(features)
            batch_lengths.append(length)
        return torch.cat(batch_features, dim=0).to(self.device), \
               torch.cat(batch_lengths, dim=0).long().to(self.device)


class TrainLoader:
    def __init__(self, features_loader, meta_artist, meta_track, artists, indexes, batch_size, crop_size,
                 augment_tracks):
        self.features_loader = features_loader
        self.meta_artist = meta_artist
        self.meta_track = meta_track
        self.artists = artists
        self.indexes = indexes
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.augment_tracks = augment_tracks

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __iter__(self):
        np.random.shuffle(self.indexes)
        for i in range(0, len(self)):
            cur_indexes = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]
            cur_indexes = np.where(np.in1d(self.meta_artist, cur_indexes))[0]
            track_ids = self.meta_track[cur_indexes]
            yield self.features_loader.load_batch(track_ids, crop_size=self.crop_size,
                                                  augment_tracks=self.augment_tracks), torch.tensor(
                self.meta_artist[cur_indexes]).long().cuda()


class TrainLoaderFactory:
    def __init__(self, features_loader):
        self.features_loader = features_loader
        self.meta_artist = self.features_loader.meta_info["artistid"].values.copy()
        self.meta_track = self.features_loader.meta_info["trackid"].values.copy()
        self.artists = np.unique(self.meta_artist)
        self.indexes = np.array(range(len(self.artists)))

    def get_loader(self, batch_size, crop_size, augment_tracks):
        return TrainLoader(self.features_loader,
                           self.meta_artist,
                           self.meta_track,
                           self.artists,
                           self.indexes,
                           batch_size,
                           crop_size,
                           augment_tracks)


class TestLoader:
    def __init__(self, features_loader, batch_size, crop_size):
        self.features_loader = features_loader
        self.meta_track = self.features_loader.meta_info['trackid'].values.copy()
        self.indexes = np.array(range(len(self.meta_track)))
        self.batch_size = batch_size
        self.crop_size = crop_size

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __iter__(self):
        for i in range(len(self)):
            cur_indexes = self.indexes[i * self.batch_size: (i + 1) * self.batch_size]
            track_ids = self.meta_track[cur_indexes]
            yield track_ids, self.features_loader.load_batch(track_ids, crop_size=self.crop_size, augment_tracks=False)
