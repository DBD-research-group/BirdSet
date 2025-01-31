import numpy as np
import warnings


class XCEventMapping:
    """
    Extracts all event_cluster into individual rows, should be used as mapping
    used on a hf dataset with batched=True (does not save the audio array, only filepath).

    Attributes
    ----------
    biggest_cluster : bool
        If set to True, the mapper focuses on the biggest cluster of events, which can be particularly useful for datasets with imbalanced event distributions.
    no_call : bool
        Indicates whether 'no-call' events should be included. In this configuration, it's set to False as the no-call samples are handled separately by the nocall_sampler.
    """

    def __init__(self, biggest_cluster: bool = True, no_call: bool = False):

        if no_call:
            warnings.warn(
                f"no_call is not working, skipping including no_calls from bird recordings"
            )
        self.biggest_cluster = biggest_cluster
        self.no_call = no_call

    def __call__(self, batch):
        # create new batch to fill: dict with name and then fill with list
        new_batch = {key: [] for key in batch.keys()}
        # new_batch["no_call_events"] = []
        # new_batch["noise_events"] = []

        for b_idx in range(len(batch.get("filepath", []))):

            detected_events = np.array(batch["detected_events"][b_idx])
            detected_cluster = np.array(batch["event_cluster"][b_idx])
            # no_call_events = self._no_call_detection(
            #     detected_events=detected_events,
            #     file_length=batch["length"][b_idx]
            # ) #!TODO LENGTH CAN BE 0???

            # noise_events = detected_events[detected_cluster==-1].tolist()

            # noise cluster not a bird, just if only 1 noise cluster is available
            if (
                not (len(detected_cluster) == 1 and detected_cluster[0] == -1)
                or len(detected_cluster) > 1
            ):
                mask = detected_cluster != -1
                detected_events = detected_events[mask]
                detected_cluster = detected_cluster[mask]

            # check if an event was found
            if len(detected_events) >= 1:
                if self.biggest_cluster:
                    values, count = np.unique(
                        detected_cluster, return_counts=True
                    )  # count clusters!
                    detected_events = detected_events[
                        detected_cluster == values[count.argmax()]
                    ]  # take the events that are most frequent as primary label: if same --> first one (0)
                    detected_cluster = detected_cluster[
                        detected_cluster == values[count.argmax()]
                    ]

                detected_events = detected_events.tolist()
                detected_cluster = detected_cluster.tolist()

                for i in range(len(detected_events)):
                    for key in new_batch.keys():
                        if key == "audio":
                            new_batch[key].append(batch["filepath"][b_idx])
                        elif key == "detected_events":
                            new_batch[key].append(detected_events[i])
                        elif key == "event_cluster":
                            new_batch[key].append([detected_cluster[i]])
                        # elif key == "no_call_events":
                        #     new_batch[key].append(no_call_events)
                        # elif key == "noise_events":
                        #     new_batch[key].append(noise_events)
                        else:
                            new_batch[key].append(batch[key][b_idx])

            else:
                for key in new_batch.keys():
                    if key == "audio":
                        new_batch[key].append(batch["filepath"][b_idx])
                    elif key == "detected_events":
                        new_batch[key].append([0, 5])
                    # elif key == "no_call_events":
                    #     new_batch[key].append(no_call_events)
                    # elif key == "noise_events":
                    #     new_batch[key].append(noise_events)
                    elif key == "event_cluster":
                        new_batch[key].append(list(detected_cluster))
                    else:
                        v = batch[key][b_idx]
                        new_batch[key].append(
                            v if v != [] else None
                        )  # could also be empty for secondary?

            # if self.no_call:
            #     if len(no_call_events) == 0:
            #         no_call_event = []
            #     else:
            #         index = random.randint(0, len(no_call_events)-1)
            #         no_call_event = no_call_events[index]
            #     for key in new_batch.keys():
            #         if key == "audio":
            #             new_batch[key].append(batch["filepath"][b_idx])
            #         elif key == "detected_events":
            #             new_batch[key].append(no_call_event)
            #         elif key == "no_call_events":
            #             new_batch[key].append(no_call_events)
            #         elif key == "noise_events":
            #             new_batch[key].append(noise_events)
            #         elif key == "event_cluster":
            #             new_batch[key].append([])
            #         elif key == "ebird_code_multilabel":
            #             new_batch[key].append([0]) # TODO this is the wrong class for multilabel, should be (0,0,...,0) this would be (1,0,0,...,0)
            #             # solve by adding a psudo class (-1) and in one hot encoding handle this class
            #         elif key == "ebird_code":
            #             new_batch[key].append(0)
            #         else:
            #             v = batch[key][b_idx]
            #             new_batch[key].append(v if v != [] else None)

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
