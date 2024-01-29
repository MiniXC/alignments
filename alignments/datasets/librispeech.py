from glob import glob
from pathlib import Path
import re
import gzip
import json
import shutil

from tqdm.auto import tqdm
from transformers.utils.hub import cached_file
import tgt

from alignments.dataset import AlignmentDataset

class LibrittsDataset(AlignmentDataset):
    def __init__(self, **kwargs):
        if "acoustic_model" not in kwargs:
            kwargs["acoustic_model"] = "english_us_arpa"
        if "g2p_model" not in kwargs:
            kwargs["g2p_model"] = "english_us_arpa"
        if "lexicon" not in kwargs:
            kwargs["lexicon"] = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
        if "textgrid_url" not in kwargs and "source_url" in kwargs and kwargs["source_url"] is not None:
            hf_url = "https://huggingface.co/datasets/cdminix/libritts-aligned/resolve/main/data/"
            if "dev-clean" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "dev_clean.tar.gz"
            elif "dev-other" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "dev_other.tar.gz"
            elif "test-clean" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "test_clean.tar.gz"
            elif "test-other" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "test_other.tar.gz"
            elif "train-clean-100" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "train_clean_100.tar.gz"
            elif "train-clean-360" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "train_clean_360.tar.gz"
            elif "train-other-500" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "train_other_500.tar.gz"
        super().__init__(**kwargs)

    def collect_data(self, directory):
        for file in Path(directory).glob("**/*.wav"):
            if file.name.startswith(".") or not file.with_suffix(".normalized.txt").is_file():
                continue
            transcript = open(file.with_suffix(".normalized.txt"), "r").read()
            if "illustration" in transcript.lower():
                transcript = transcript.replace("Illustration:", " ")
            transcript = re.sub("[\[\]\(\)-]", " ", transcript)
            transcript = re.sub("\s+", " ", transcript)
            transcript = re.sub(r"'(\w)'", "\g<1>", transcript)
            yield {
                "path": file,
                "speaker": file.parent.parent.name,
                "transcript": transcript.upper(),
            }

class LibrittsRDataset(AlignmentDataset):
    def __init__(self, **kwargs):
        if "acoustic_model" not in kwargs:
            kwargs["acoustic_model"] = "english_us_arpa"
        if "g2p_model" not in kwargs:
            kwargs["g2p_model"] = "english_us_arpa"
        if "lexicon" not in kwargs:
            kwargs["lexicon"] = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
        if "textgrid_url" not in kwargs and "source_url" in kwargs and kwargs["source_url"] is not None:
            hf_url = "https://huggingface.co/datasets/cdminix/libritts-r-aligned/resolve/main/data/"
            if "dev_clean" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "dev_clean.tar.gz"
            elif "dev_other" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "dev_other.tar.gz"
            elif "test_clean" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "test_clean.tar.gz"
            elif "test_other" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "test_other.tar.gz"
            elif "train_clean_100" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "train_clean_100.tar.gz"
            elif "train_clean_360" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "train_clean_360.tar.gz"
            elif "train_other_500" in kwargs["source_url"]:
                kwargs["textgrid_url"] = hf_url + "train_other_500.tar.gz"
        super().__init__(**kwargs)

    def collect_data(self, directory):
        for file in Path(directory).glob("**/*.wav"):
            if file.name.startswith(".") or not file.with_suffix(".normalized.txt").is_file():
                continue
            if file.name == '1092_134562_000013_000004.wav':
                # avoid broken file
                continue 
            transcript = open(file.with_suffix(".normalized.txt"), "r").read()
            if "illustration" in transcript.lower():
                transcript = transcript.replace("Illustration:", " ")
            transcript = re.sub("[\[\]\(\)-]", " ", transcript)
            transcript = re.sub("\s+", " ", transcript)
            transcript = re.sub(r"'(\w)'", "\g<1>", transcript)
            yield {
                "path": file,
                "speaker": file.parent.parent.name,
                "transcript": transcript.upper(),
            }


class LibriHeavyDataset(AlignmentDataset):
    def __init__(self, jsonl_path, overwrite_source_dir=True, **kwargs):
        if "acoustic_model" not in kwargs:
            kwargs["acoustic_model"] = "english_us_arpa"
        if "g2p_model" not in kwargs:
            kwargs["g2p_model"] = "english_us_arpa"
        if "lexicon" not in kwargs:
            kwargs["lexicon"] = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
        if "source_url" in kwargs and kwargs["source_url"] is not None:
            raise ValueError("source url not supported for libriheavy,\
            please use the preprocessing at https://github.com/k2-fsa/libriheavy")
        if "textgrid_url" in kwargs:
            raise ValueError("textgrid url not supported for libriheavy (yet)")
        # prepare source_directory, if it doesn't exist yet
        if overwrite_source_dir and Path(kwargs["source_directory"]).exists():
            shutil.rmtree(kwargs["source_directory"])
        if not Path(kwargs["source_directory"]).exists():
            Path(kwargs["source_directory"]).mkdir(parents=True)
            # construct the textgrid files from the jsonl
            recording_set = set()
            total_recording_duration = 0
            with gzip.open(jsonl_path, mode='rt', encoding="utf-8") as gz_file:
                i = 0
                skips = 0
                seconds = 0
                seconds_all = 0
                previous_source = None
                for line in tqdm(gz_file):
                    i += 1
                    data = json.loads(line)
                    if data["recording"]["id"] not in recording_set:
                        recording_set.add(data["recording"]["id"])
                        total_recording_duration += data["recording"]["duration"]
                    current_source = data["recording"]["sources"][0]["source"]
                    if len(data["supervisions"]) > 1:
                        raise ValueError("multiple supervisions not supported")
                    speaker = data["supervisions"][0]["speaker"]
                    if current_source != previous_source:
                        if previous_source is not None:
                            Path(tgt_path).parent.mkdir(parents=True, exist_ok=True)
                            tgt.io.write_to_file(tgt_file, tgt_path)
                            # symlink flac file
                            if not Path(previous_source).exists():
                                Path(tgt_path).resolve().with_suffix(".flac").symlink_to(Path(previous_source).resolve())
                        tgt_path = (Path(kwargs["source_directory"]) / speaker / Path(current_source).name).with_suffix(".TextGrid")
                        if tgt_path.exists():
                            tgt_file = tgt.io.read_textgrid(tgt_path)
                        else:
                            tgt_file = tgt.core.TextGrid(tgt_path)
                        if not tgt_file.has_tier(speaker):
                            tgt_file.add_tier(
                                tgt.core.IntervalTier(
                                    float(data["start"]), 
                                    float(data["start"]) + float(data["duration"]),
                                    speaker
                                )
                            )
                        previous_source = current_source
                    speaker_tier = tgt_file.get_tier_by_name(speaker)
                    try:
                        speaker_tier.add_interval(
                            tgt.core.Interval(
                                float(data["start"]), 
                                float(data["start"]) + float(data["duration"]),
                                data["supervisions"][0]["custom"]["texts"][0]
                            )
                        )
                        seconds_all += float(data["duration"])
                        seconds += float(data["duration"])
                    except ValueError as e:
                        # replace the interval if the new one is shorter
                        interval = speaker_tier.get_annotations_between_timepoints(
                            float(data["start"]),
                            float(data["start"]) + float(data["duration"]),
                            left_overlap=True,
                            right_overlap=True
                        )[0]
                        old_duration = interval.end_time - interval.start_time
                        seconds_all -= old_duration
                        old_start, old_end = interval.start_time, interval.end_time
                        new_start, new_end = float(data["start"]), float(data["start"]) + float(data["duration"])
                        seconds_all += max(new_end, old_end) - min(new_start, old_start)
                        new_duration = float(data["duration"])
                        if new_duration > old_duration:
                            speaker_tier.delete_annotations_between_timepoints(
                                float(data["start"]),
                                float(data["start"]) + float(data["duration"]),
                                left_overlap=True,
                                right_overlap=True
                            )
                            speaker_tier.add_interval(
                                tgt.core.Interval(
                                    float(data["start"]), 
                                    float(data["start"]) + float(data["duration"]),
                                    data["supervisions"][0]["custom"]["texts"][0]
                                )
                            )
                            seconds += float(data["duration"]) - old_duration
                        skips += 1
        print(f"skipped {round(skips / i * 100, 2)}% of the lines due to overlap")
        print(f"added {round(seconds / 3600, 2)} hours of audio")
        print(f"added {round(seconds_all / 3600, 2)} hours of audio in theory")
        print(f"added {round(total_recording_duration / 3600, 2)} hours of audio in total")
        super().__init__(**kwargs)

    def collect_data(self, file):
        pass
