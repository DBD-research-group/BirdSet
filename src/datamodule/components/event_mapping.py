import numpy as np
import random

class Mapper:
    def __call__(self, batch):
        return batch

class FixedSizeMapper(Mapper):
    def __init__(self, sequence_length: int = 90112 ) -> None:
        self.sequence_length = sequence_length
    
    def __call__(self, batch):
        '''
        sets the size of the audio array to a fixed size
        '''
        for b_idx in range(len(batch["audio"])):
            if batch["audio"][b_idx]["array"].shape[0] > self.sequence_length:
                # select random part of the audio array
                start = random.randint(0, batch["audio"][b_idx]["array"].shape[0] - self.sequence_length)
                batch["audio"][b_idx]["array"] = batch["audio"][b_idx]["array"][start:start + self.sequence_length]
            else:
                batch["audio"][b_idx]["array"] = np.pad(
                    batch["audio"][b_idx]["array"],
                    (0, self.sequence_length - batch["audio"][b_idx]["array"].shape[0]),
                    mode="constant",
                )
        return batch




class XCEventMapping(Mapper):
    """extracts all event_cluster into individual rows, should be used as mapping
     used on a hf dataset with batched=True (does not save the audio array, only filepath)

     with_noise_cluster: weather to include cluster that are marker as noise from event detection
            IF True: when all cluster are noise cluster, cluster will be deleted and audio will be included once
     biggest_cluster: use only samples from the biggest cluster, which might be the primary bird
     only_one: choose one of the events at random per audio"""
    def __init__(
            self, 
            biggest_cluster: bool = True, 
            event_limit: int = 5,
            no_call: bool = True):

        self.event_limit = event_limit
        self.biggest_cluster = biggest_cluster
        self.no_call = no_call

    def __call__(self, batch):
        # create new batch to fill: dict with name and then fill with list
        new_batch = {key: [] for key in batch.keys()} 
        new_batch["no_call_events"] = []
        new_batch["noise_events"] = [] 

        for b_idx in range(len(batch.get('filepath', []))):

            detected_events = np.array(batch["detected_events"][b_idx])
            detected_cluster = np.array(batch["event_cluster"][b_idx])
            no_call_events = self._no_call_detection(
                detected_events=detected_events, 
                file_length=batch["length"][b_idx]
            ) #!TODO LENGTH CAN BE 0???

            noise_events = detected_events[detected_cluster==-1].tolist()

            # noise cluster not a bird, just if only 1 noise cluster is available
            if not (len(detected_cluster) == 1 and detected_cluster[0] == -1) or len(detected_cluster) > 1:
                mask = detected_cluster != -1
                detected_events = detected_events[mask]
                detected_cluster = detected_cluster[mask]

            # check if an event was found 
            if len(detected_events) >= 1: 
                if self.biggest_cluster:
                    values, count = np.unique(detected_cluster, return_counts=True) # count clusters!
                    detected_events = detected_events[detected_cluster == values[count.argmax()]]# take the events that are most frequent as primary label: if same --> first one (0)
                    detected_cluster = detected_cluster[detected_cluster == values[count.argmax()]]
                    #batch["detected_events"][b_idx] = events[cluster == values[count.argmax()]].tolist()

                # sample # n events 
                n_detected_events = len(detected_events)

                if self.event_limit == 1: # move to decoding
                    index = random.randint(0, n_detected_events-1)
                    detected_events = [(detected_events[index]).tolist()]
                    detected_cluster = [detected_cluster[index]]

                elif self.event_limit == None:
                    detected_events = detected_events.tolist()
                    detected_cluster = detected_cluster.tolist()

                else:
                    if n_detected_events < self.event_limit:
                        detected_cluster = detected_cluster.tolist()
                        detected_events = detected_events.tolist()
                    else:
                        indices = random.sample(range(n_detected_events), self.event_limit)
                        detected_events = [detected_events[i].tolist() for i in indices]
                        detected_cluster = [detected_cluster[i].tolist() for i in indices]

                # add data for all detected events, but limit it to self.event_limit! 
                for i in range(len(detected_events)):
                    for key in new_batch.keys():
                        if key == "audio":
                            new_batch[key].append(batch["filepath"][b_idx])
                        elif key == "detected_events":
                            new_batch[key].append(detected_events[i])
                        elif key == "event_cluster":
                            new_batch[key].append([(detected_cluster[i])])
                        elif key == "no_call_events":
                            new_batch[key].append(no_call_events)
                        elif key == "noise_events":
                            new_batch[key].append(noise_events)
                        else:
                            new_batch[key].append(batch[key][b_idx])
                        
            else: 
                file_length = batch["length"][b_idx]
                for key in new_batch.keys():
                    if key == "audio":
                        new_batch[key].append(batch["filepath"][b_idx])      
                        #double 5 seconds no_call vs new_batch!                    
                    # if no event cluster is found --> first 5 seconds of audio!
                    elif key == "detected_events": 
                        if file_length >= 5:
                             ## 5 only if longer than 5!!
                            new_batch[key].append([0,5])
                        else:
                            new_batch[key].append([0,file_length])
                    elif key == "no_call_events":
                        #start no_call @ 5 because we simulated the detected event! needs to be fixec
                        new_batch[key].append(no_call_events)
                    elif key == "noise_events":
                        new_batch[key].append(noise_events)    
                    elif key == "event_cluster":
                        new_batch[key].append(list(detected_cluster))    
                    else:
                        v = batch[key][b_idx]
                        new_batch[key].append(v if v != [] else None) # could also be empty for secondary?
            
            if self.no_call:
                if len(no_call_events) == 0:
                    no_call_event = []
                else:
                    index = random.randint(0, len(no_call_events)-1)
                    no_call_event = no_call_events[index]
                for key in new_batch.keys():
                    if key == "audio":
                        new_batch[key].append(batch["filepath"][b_idx])   
                    elif key == "detected_events": 
                        new_batch[key].append(no_call_event)
                    elif key == "no_call_events":
                        new_batch[key].append(no_call_events)
                    elif key == "noise_events":
                        new_batch[key].append(noise_events)    
                    elif key == "event_cluster":
                        new_batch[key].append([])
                    elif key == "ebird_code_multilabel":
                        new_batch[key].append([0]) # add no_call = 0
                    elif key == "ebird_code":
                        new_batch[key].append(0) 
                    else:
                        v = batch[key][b_idx]
                        new_batch[key].append(v if v != [] else None) 

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

        

# class XCEventMapping:
#     def __init__(self, with_noise_cluster=False, biggest_cluster=False, only_one=False, event_limit=5):
#         self.event_limit = None
#         self.biggest_cluster = biggest_cluster
#         self.no_call = True

#     def __call__(self, batch):
#         new_batch = {key: [] for key in batch.keys()}
#         new_batch["no_call_events"] = []
#         new_batch["noise_events"] = []

#         for b_idx in range(len(batch.get('filepath', []))):
#             detected_events = np.array(batch["detected_events"][b_idx])
#             detected_cluster = np.array(batch["event_cluster"][b_idx])
#             no_call_events = self._no_call_detection(detected_events=detected_events, file_length=batch["length"][b_idx])
#             noise_events = detected_events[detected_cluster == -1].tolist()

#             if not (len(detected_cluster) == 1 and detected_cluster[0] == -1) or len(detected_cluster) > 1:
#                 mask = detected_cluster != -1
#                 detected_events = detected_events[mask]
#                 detected_cluster = detected_cluster[mask]

#             if len(detected_events) >= 1:
#                 if self.biggest_cluster:
#                     values, count = np.unique(detected_cluster, return_counts=True)
#                     detected_events = detected_events[detected_cluster == values[count.argmax()]]
#                     detected_cluster = detected_cluster[detected_cluster == values[count.argmax()]]

#                 n_detected_events = len(detected_events)

#                 if self.event_limit == 1:
#                     index = random.randint(0, n_detected_events - 1)
#                     detected_events = [detected_events[index].tolist()]
#                     detected_cluster = [detected_cluster[index]]

#                 elif self.event_limit is None:
#                     detected_events = detected_events.tolist()
#                     detected_cluster = detected_cluster.tolist()

#                 else:
#                     if n_detected_events < self.event_limit:
#                         detected_cluster = detected_cluster.tolist()
#                         detected_events = detected_events.tolist()
#                     else:
#                         indices = random.sample(range(n_detected_events), self.event_limit)
#                         detected_events = [detected_events[i].tolist() for i in indices]
#                         detected_cluster = [detected_cluster[i] for i in indices]

#                 for i in range(len(detected_events)):
#                     for key in new_batch.keys():
#                         if key == "audio":
#                             new_batch[key].append(batch["filepath"][b_idx])
#                         elif key == "detected_events":
#                             new_batch[key].append(detected_events[i])
#                         elif key == "event_cluster":
#                             new_batch[key].append([detected_cluster[i]])
#                         elif key == "no_call_events":
#                             new_batch[key].append(no_call_events)
#                         elif key == "noise_events":
#                             new_batch[key].append(noise_events)
#                         else:
#                             new_batch[key].append(batch[key][b_idx])

#             else:
#                 for key in new_batch.keys():
#                     if key == "audio":
#                         new_batch[key].append(batch["filepath"][b_idx])
#                     elif key == "detected_events":
#                         new_batch[key].append([0, 5])
#                     elif key == "no_call_events":
#                         no_call_events[0][0] = 5
#                         new_batch[key].append(no_call_events)
#                     elif key == "noise_events":
#                         new_batch[key].append(noise_events)
#                     elif key == "event_cluster":
#                         new_batch[key].append(list(detected_cluster))
#                     else:
#                         v = batch[key][b_idx]
#                         new_batch[key].append(v if v != [] else None)

#             if self.no_call:
#                 index = random.randint(0, len(no_call_events) - 1)
#                 no_call_event = no_call_events[index]
#                 if len(no_call_event) == 2:
#                     no_call_event = [no_call_event[0], min(no_call_event[0] + 5, no_call_event[1])]
#                 else:
#                     no_call_event = [0, 5]

#                 for key in new_batch.keys():
#                     if key == "audio":
#                         new_batch[key].append(batch["filepath"][b_idx])
#                     elif key in ["detected_events", "no_call_events", "noise_events"]:
#                         new_batch[key].append(no_call_event)
#                     elif key == "event_cluster":
#                         new_batch[key].append([])  # Append an empty list for consistency
#                     elif key in ["ebird_code_multilabel", "ebird_code"]:
#                         new_batch[key].append(0)  # Assuming 0 is the default value
#                     else:
#                         v = batch[key][b_idx]
#                         new_batch[key].append(v if isinstance(v, list) else [v])  # Wrap non-lists in a list

#         return new_batch

#     def _no_call_detection(self, detected_events, file_length):
#         no_event_periods = []
#         last_end_time = 0
#         for start, end in detected_events:
#             if last_end_time < start:
#                 no_event_periods.append([last_end_time, start])
#             last_end_time = end
#         if last_end_time < file_length:
#             no_event_periods.append([last_end_time, file_length])
#         return no_event_periods



