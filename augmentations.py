import random


# shape = (channels, timesteps)

def augment_null_timestamps(track, max_count, probability):
    for _ in range(max_count):
        if random.random() < probability:
            random_length = random.randint(2, 5)
            random_start = random.randint(0, track.shape[1] - random_length)
            track[:, random_start: random_start + random_length] = 0
    return track


def augment_null_channels(track, max_count, probability):
    for _ in range(max_count):
        if random.random() < probability:
            random_len = random.randint(2, 5)
            random_start = random.randint(0, track.shape[0] - random_len)
            track[random_start: random_start + random_len, :] = 0
    return track


def augment_swap_timesteps(track, max_count, probability):
    for _ in range(max_count):
        if random.random() < probability:
            random_len = random.randint(2, 10)
            random_start = random.randint(0, track.shape[1] - random_len)
            random_start_2 = random.randint(0, track.shape[1] - random_len)
            track[:, random_start: random_start + random_len] = track[:,
                                                                random_start_2: random_start_2 + random_len]
    return track


def augment_swap_channels(track, max_count, probability):
    for _ in range(max_count):
        if random.random() < probability:
            random_len = random.randint(2, 5)
            random_start = random.randint(0, track.shape[0] - random_len)
            random_start_2 = random.randint(0, track.shape[0] - random_len)
            track[random_start: random_start + random_len, :] = track[
                                                                random_start_2: random_start_2 + random_len, :]
    return track
