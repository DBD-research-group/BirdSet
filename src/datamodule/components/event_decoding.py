import soundfile as sf
import librosa
import random
import soundfile as sf
import librosa

class EventDecoding:
    def __init__(self, min_len=1, max_len=None, sampling_rate=None, extension_time=7, extracted_interval=5):
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
        event_duration = round(end - start, 2)

        if event_duration < self.extension_time:
            side_extension_time = (self.extension_time - event_duration) / 2
            new_start_time = round(max(0, start - side_extension_time))
            new_end_time = round(min(total_duration, end + side_extension_time))

            if new_end_time - new_start_time < self.extension_time:
                if new_start_time == 0:
                    new_end_time = min(self.extension_time, total_duration)
                elif new_end_time == total_duration: 
                    new_start_time = max(0, total_duration - self.extension_time)
        
        else: # longer than extraction time
            new_start_time = start
            new_end_time = end
        
        # select a random interval of 5 seconds
        max_start_interval = new_end_time - self.extracted_interval
        random_start = random.uniform(new_start_time, max_start_interval)
        random_end = random_start + self.extracted_interval

        return random_start, random_end
    

    def __call__(self, batch):
        audios, srs = [], []
        for b_idx in range(len(batch.get("filepath", []))):
            file_info = sf.info(batch["filepath"][b_idx])
            sr = file_info.samplerate
            duration = file_info.duration

            if batch["detected_events"][b_idx]: #only for train data, not test
                start, end = batch["detected_events"][b_idx]
                if self.extension_time:
                    #time shifting
                    start, end = self._time_shifting(start, end, duration)
            else:
                start, end = batch["start_time"][b_idx], batch["end_time"][b_idx]
            audio, sr = self._load_audio(batch["filepath"][b_idx], start, end, sr)
            audios.append(audio)
            srs.append(sr)
        if batch.get("filepath", None):
            batch["audio"] = [{"path": path, "array": audio, "samplerate": sr} for audio, path, sr in zip(audios, batch["filepath"], srs)]
        return batch
    
    
