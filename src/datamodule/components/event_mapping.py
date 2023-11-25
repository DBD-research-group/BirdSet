import numpy as np
import random


class EventMapping:
    """extracts all event_cluster into individual rows, should be used as mapping
     used on a hf dataset with batched=True (does not save the audio array, only filepath)

     with_noise_cluster: weather to include cluster that are marker as noise from event detection
            IF True: when all cluster are noise cluster, cluster will be deleted and audio will be included once
     biggest_cluster: use only samples from the biggest cluster, which might be the primary bird
     only_one: choose one of the events at random per audio"""
    def __init__(self, with_noise_cluster: bool = False, biggest_cluster: bool = False, only_one: bool = False):
        self.with_noise_cluster = with_noise_cluster
        self.biggest_cluster = biggest_cluster
        self.only_one = only_one

    def __call__(self, batch):
        new_batch = {key: [] for key in batch.keys()}

        for b_idx in range(len(batch.get('filepath', []))):
            if not self.with_noise_cluster:
                events = np.array(batch["detected_events"][b_idx])
                cluster = np.array(batch["event_cluster"][b_idx])
                batch["detected_events"][b_idx] = events[cluster != -1].tolist()
                batch["event_cluster"][b_idx] = cluster[cluster != -1].tolist()
            if len(batch['detected_events'][b_idx]) >= 1:
                if self.biggest_cluster:
                    events = np.array(batch["detected_events"][b_idx])
                    cluster = np.array(batch["event_cluster"][b_idx])
                    values, count = np.unique(cluster, return_counts=True)
                    batch["detected_events"][b_idx] = events[cluster == values[count.argmax()]].tolist()
                    batch["event_cluster"][b_idx] = cluster[cluster == values[count.argmax()]].tolist()
                if self.only_one:
                    r = random.randint(0, len(batch["event_cluster"][b_idx]) - 1)
                    batch["detected_events"][b_idx] = [batch["detected_events"][b_idx][r]]
                    batch["event_cluster"][b_idx] = [batch["event_cluster"][b_idx][r]]

                for i in range(len(batch['detected_events'][b_idx])):
                    for key in new_batch.keys():
                        if key == "audio":
                            new_batch[key].append(batch["filepath"][b_idx])
                        elif key == "detected_events" or key == "event_cluster":
                            new_batch[key].append(batch[key][b_idx][i])
                        else:
                            new_batch[key].append(batch[key][b_idx])
            else:
                for key in new_batch.keys():
                    if key == "audio":
                        new_batch[key].append(batch["filepath"][b_idx])
                    else:
                        v = batch[key][b_idx]
                        new_batch[key].append(v if v != [] else None)
        return new_batch


class EventSegmenting:
    """Segments the audio to only contain the audio part of the detected event
    used on a hf dataset, in set_transform, loads the audio and returns the audio array"""
    def __init__(self):
        pass

    def __call__(self, batch):
        for b_idx in range(len(batch["audio"])):
            if batch["detected_events"][b_idx]:
                start, stop = batch["detected_events"][b_idx]
                sr = batch["audio"][b_idx]["sampling_rate"]
                batch["audio"][b_idx]["array"] = batch["audio"][b_idx]["array"][int(start * sr):int(stop * sr)]
        return batch


