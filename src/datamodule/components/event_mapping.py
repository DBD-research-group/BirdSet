import numpy as np
import random

# class XCEventMapping:
#     """extracts all event_cluster into individual rows, should be used as mapping
#      used on a hf dataset with batched=True (does not save the audio array, only filepath)

#      with_noise_cluster: weather to include cluster that are marker as noise from event detection
#             IF True: when all cluster are noise cluster, cluster will be deleted and audio will be included once
#      biggest_cluster: use only samples from the biggest cluster, which might be the primary bird
#      only_one: choose one of the events at random per audio"""
#     def __init__(
#             self, 
#             with_noise_cluster: bool = False, 
#             biggest_cluster: bool = False, 
#             only_one: bool = False,
#             limit: int = 5):
        
#         self.with_noise_cluster = with_noise_cluster
#         self.biggest_cluster = biggest_cluster
#         self.only_one = only_one
#         self.limit = limit

#     def __call__(self, batch):
#         new_batch = {key: [] for key in batch.keys()}

#         for b_idx in range(len(batch.get('filepath', []))):
#             if not self.with_noise_cluster:
#                 events = np.array(batch["detected_events"][b_idx])
#                 cluster = np.array(batch["event_cluster"][b_idx])
#                 batch["detected_events"][b_idx] = events[cluster != -1].tolist()
#                 batch["event_cluster"][b_idx] = cluster[cluster != -1].tolist()

#             if len(batch['detected_events'][b_idx]) >= 1:
#                 if self.biggest_cluster:
#                     events = np.array(batch["detected_events"][b_idx])
#                     cluster = np.array(batch["event_cluster"][b_idx])
#                     values, count = np.unique(cluster, return_counts=True)
#                     batch["detected_events"][b_idx] = events[cluster == values[count.argmax()]].tolist()
#                     batch["event_cluster"][b_idx] = cluster[cluster == values[count.argmax()]].tolist()
#                 if self.only_one:
#                     r = random.randint(0, len(batch["event_cluster"][b_idx]) - 1)
#                     batch["detected_events"][b_idx] = [batch["detected_events"][b_idx][r]]
#                     batch["event_cluster"][b_idx] = [batch["event_cluster"][b_idx][r]]

#                 for i in range(len(batch['detected_events'][b_idx])):
#                     for key in new_batch.keys():
#                         if key == "audio":
#                             new_batch[key].append(batch["filepath"][b_idx])
#                         elif key == "detected_events" or key == "event_cluster":
#                             new_batch[key].append(batch[key][b_idx][i])
#                         else:
#                             new_batch[key].append(batch[key][b_idx])
#             else:
#                 for key in new_batch.keys():
#                     if key == "audio":
#                         new_batch[key].append(batch["filepath"][b_idx])
#                     else:
#                         v = batch[key][b_idx]
#                         new_batch[key].append(v if v != [] else None)
#         return new_batch
    
class XCEventMapping:
    """extracts all event_cluster into individual rows, should be used as mapping
     used on a hf dataset with batched=True (does not save the audio array, only filepath)

     with_noise_cluster: weather to include cluster that are marker as noise from event detection
            IF True: when all cluster are noise cluster, cluster will be deleted and audio will be included once
     biggest_cluster: use only samples from the biggest cluster, which might be the primary bird
     only_one: choose one of the events at random per audio"""
    def __init__(
            self, 
            with_noise_cluster: bool = False, 
            biggest_cluster: bool = False, 
            only_one: bool = False,
            event_limit: int = 5):
        
        self.with_noise_cluster = with_noise_cluster
        self.biggest_cluster = biggest_cluster
        self.only_one = only_one
        self.event_limit = event_limit

        self.no_call = True

    def __call__(self, batch):
        # create new batch to fill: dict with name and then fill with list
        new_batch = {key: [] for key in batch.keys()} 

        for b_idx in range(len(batch.get('filepath', []))):

            detected_events = np.array(batch["detected_events"][b_idx])
            detected_cluster = np.array(batch["event_cluster"][b_idx])
            no_call_events = np.array(
                self._no_call_detection(
                    detected_events=detected_events, 
                    file_length=batch["length"][b_idx]
                    )
                )
            noise_events = detected_events[detected_cluster==-1]

            if not self.with_noise_cluster:
                # remove all noise events with -1
                detected_events = detected_events[detected_cluster != -1]
                detected_cluster = detected_cluster[detected_cluster != -1]

            # check if an event was found 
            if len(detected_events) >= 1: 

                if self.biggest_cluster:
                    values, count = np.unique(detected_cluster, return_counts=True) # count clusters!
                    detected_events = detected_events[detected_cluster == values[count.argmax()]]# take the events that are most frequent as primary label: if same --> first one (0)
                    detected_cluster = detected_cluster[detected_cluster == values[count.argmax()]]
                    #batch["detected_events"][b_idx] = events[cluster == values[count.argmax()]].tolist()

                # limit the detected events to 1    
                if self.only_one:
                    r = random.randint(0, len(detected_cluster) - 1) # move this to decoding, not here!
                    detected_events = [list(detected_events[r])]
                    detected_cluster = [detected_cluster[r]]

                # add data for all detected events, but limit it to self.event_limit! 
                for i in range(min(self.event_limit, len(detected_events))):
                    for key in new_batch.keys():
                        if key == "audio":
                            new_batch[key].append(batch["filepath"][b_idx])
                        elif key == "detected_events":
                            new_batch[key].append(detected_events[i])
                        elif key == "event_cluster":
                            new_batch[key].append(detected_cluster[i])
                        else:
                            new_batch[key].append(batch[key][b_idx])

                if self.no_call:
                    r = random.randint(0, len(no_call_events-1))
                    no_call_events = 
                        
                        #for i in range
            # when no event cluster is found
            else: 
                for key in new_batch.keys():
                    if key == "audio":
                        new_batch[key].append(batch["filepath"][b_idx])      
                    
                    # if no event cluster is found --> first 5 seconds of audio!
                    elif key == "detected_events": 
                        new_batch[key].append([0,5])        
                    else:
                        v = batch[key][b_idx]
                        new_batch[key].append(v if v != [] else None) # could also be empty for secondary?
        return new_batch
    
    def _no_call_detection(self, detected_events, file_length):
        no_event_periods = []
        last_end_time = 0

        for start, end in detected_events: 
            if last_end_time < start: 
                no_event_periods.append([last_end_time, start])
            last_end_time = end
        
        if last_end_time < file_length:
            no_event_periods.append([last_end_time, file_length])
        
        return no_event_periods

        

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


