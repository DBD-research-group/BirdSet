class EventMapping:
    """extracts all event_cluster into individual rows, should be used as mapping
     used on a hf dataset with batched=True (does not save the audio array, only filepath)

     with_no_call: weather to include cluster that are marker as noise from event detection,
     biggest_cluster: use only samples from the biggest cluster, which might be the primary bird
     only_one: choose one of the events at random per audio"""
    def __init__(self, with_no_call: bool = False, biggest_cluster: bool = False, only_one: bool = False):
        self.with_no_call = with_no_call
        if biggest_cluster or only_one:
            raise NotImplementedError("not yet implemented")

    def __call__(self, batch):
        new_batch = {key: [] for key in batch.keys()}
        for b_idx in range(len(batch['filepath'])):
            if len(batch['detected_events'][b_idx]) >= 1:
                for i in range(len(batch['detected_events'][b_idx])):
                    if not self.with_no_call and batch["event_cluster"][b_idx][i] != -1:
                        continue
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
            start, stop = batch["detected_events"][b_idx]
            sr = batch["audio"][b_idx]["sampling_rate"]
            batch["audio"][b_idx]["array"] = batch["audio"][b_idx]["array"][int(start * sr):int(stop * sr)]
        return batch


