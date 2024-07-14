import random
import soundfile as sf
import librosa


class EventDecoding:
    """
    A class used to configure the decoding of audio events.

    Attributes
    ----------
    min_len : float
        Determines the minimum duration (in seconds) of the audio segments after decoding. This constraint ensures that each processed audio segment is of a suitable length for the model.
    max_len : float
        Determines the maximum duration (in seconds) of the audio segments after decoding. This constraint ensures that each processed audio segment is of a suitable length for the model.
    sampling_rate : int
        Defines the sampling rate to which the audio should be resampled. This standardizes the input data's sampling rate, making it consistent for model processing.
    extension_time : float
        Refers to the time (in seconds) by which the duration of an audio event is extended. This parameter is crucial for ensuring that shorter audio events are sufficiently long for the model to process effectively.
    extracted_interval : float
        Denotes the fixed duration (in seconds) of the audio segment that is randomly extracted from the extended audio event.
    """
    def __init__(self,
                 min_len: float = 1,
                 max_len: float = 5,
                 sampling_rate: int = 32000,
                 extension_time: float = 8,
                 extracted_interval: float = 5):
        self.min_len = min_len # in seconds
        self.max_len = max_len
        self.sampling_rate = sampling_rate
        self.extension_time = extension_time
        self.extracted_interval = extracted_interval

    def _load_audio(self, path, start=None, end=None, sr=None):
        if start is not None and end is not None:
            if end - start < self.min_len:  # TODO: improve, eg. edge cases, more dynamic loading
                end = start + self.min_len
            if self.max_len and end - start > self.max_len:
                end = start + self.max_len
            start, end = int(start * sr), int(end * sr)
        if not end:
            end = int(self.max_len * sr)

        audio, sr = sf.read(path, start=start, stop=end)

        if audio.ndim != 1:
            audio = audio.swapaxes(1, 0)
            audio = librosa.to_mono(audio)
        if sr != self.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
            sr = self.sampling_rate
        return audio, sr

    def _time_shifting(self, start, end, total_duration):
        event_duration = end - start

        if event_duration < self.extension_time:
            side_extension_time = (self.extension_time - event_duration) / 2
            new_start_time = max(0, start - side_extension_time)
            new_end_time = min(total_duration, end + side_extension_time)

            if new_end_time - new_start_time < self.extension_time:
                if new_start_time == 0:
                    new_end_time = min(self.extension_time, total_duration)
                elif new_end_time == total_duration:
                    new_start_time = max(0, total_duration - self.extension_time)

        else:  # longer than extraction time
            new_start_time = start
            new_end_time = end

        # Ensure max_start_interval is non-negative
        max_start_interval = max(0, new_end_time - self.extracted_interval)
        random_start = random.uniform(new_start_time, max_start_interval)
        random_end = random_start + self.extracted_interval
        return random_start, random_end

    def __call__(self, batch):
        """
        Decodes an audio from a given batch by prioritizing load from the detected events.
        If no detected_events are given from start_time and end_time else loads the whole audio.
        possible to load audio by only start_time or end_time missing is 0 or len(audio) respectively.
        If detected_events are used, extends the time and chooses a random subpart of length extracted_interval.
        Expects batch to have following entries:
            - filepath, list of audio files loadable by soundfile, else nothing is loaded
        optional entries:
            - detected_events, list of (start, end)-time-tuple
            - start_time, start timestamp
            - end_time, end timestamp
        """
        audios, srs = [], []
        batch_len = len(batch.get("filepath", []))
        for b_idx in range(batch_len):
            file_info = sf.info(batch["filepath"][b_idx])
            sr = file_info.samplerate
            duration = file_info.duration

            if batch.get("detected_events", []) and batch["detected_events"][b_idx]:
                start, end = batch["detected_events"][b_idx]
                if self.extension_time:
                    #time shifting
                    start, end = self._time_shifting(start, end, duration)
            elif (batch.get("start_time", []) or batch.get("end_time", [])) and (batch["start_time"][b_idx] or batch["end_time"][b_idx]):
                start, end = batch["start_time"][b_idx], batch["end_time"][b_idx]
            else:
                start, end = None, None
            audio, sr = self._load_audio(batch["filepath"][b_idx], start, end, sr)
            audios.append(audio)
            srs.append(sr)
        if batch.get("filepath", None):
            batch["audio"] = [{"path": path, "array": audio, "samplerate": sr} for audio, path, sr in zip(audios, batch["filepath"], srs)]
        return batch
    
    
