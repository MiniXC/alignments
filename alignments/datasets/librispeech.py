from glob import glob
from pathlib import Path
import re
import gzip
import json
import shutil

from tqdm.auto import tqdm
from transformers.utils.hub import cached_file
import tgt
import torchaudio
import torch
import librosa

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
            with gzip.open(jsonl_path, mode='rt', encoding="utf-8") as gz_file:
                i = 0
                seconds = 0
                skips = 0
                source_dict = {}
                previous_source = None
                prev_int_hours = 0
                for line in tqdm(gz_file):
                    i += 1
                    data = json.loads(line)
                    current_source = Path(jsonl_path).parent / Path(data["recording"]["sources"][0]["source"])
                    if current_source not in source_dict:
                        source_dict[current_source] = []
                    source_dict[current_source].append([
                        len(source_dict[current_source]) + 1,
                        data["start"],
                        data["duration"],
                        data["supervisions"][0]["speaker"],
                        data["supervisions"][0]["custom"]["texts"][0],
                    ])
                    if len(data["supervisions"]) > 1:
                        raise ValueError("multiple supervisions not supported")
                    # tgt_path_lab = (Path(kwargs["source_directory"]) / speaker / (Path(current_source).name + f"_{source_dict[current_source][-1][0]}")).with_suffix(".lab")
                    # tgt_path_flac = tgt_path_lab.with_suffix(".flac")
                    # transcript = data["supervisions"][0]["custom"]["texts"][0]
                    # start_time = float(data["start"])
                    # end_time = float(data["start"]) + float(data["duration"])
                    # # save transcript to lab file
                    # tgt_path_lab.parent.mkdir(parents=True, exist_ok=True)
                    # with open(tgt_path_lab, "w") as f:
                    #     f.write(transcript)
                    if current_source != previous_source and previous_source is not None:
                        # save transcripts to lab files
                        audio, sample_rate = torchaudio.load(current_source)
                        for source in source_dict[previous_source]:
                            new_audio = audio[:, int(source[1] * sample_rate):int((source[1] + source[2]) * sample_rate)]
                            if new_audio.shape[1] > 0:
                                tgt_path_lab = (Path(kwargs["source_directory"]) / source[3] / (Path(previous_source).name.replace(".flac", "") + f"_{source[0]}")).with_suffix(".lab")
                                tgt_path_flac = tgt_path_lab.with_suffix(".flac")
                                tgt_path_lab.parent.mkdir(parents=True, exist_ok=True)
                                with open(tgt_path_lab, "w") as f:
                                    f.write(source[4])
                                # save audio to flac files
                                if not tgt_path_flac.is_file():
                                    torchaudio.save(tgt_path_flac, new_audio, sample_rate)
                                else:
                                    print(f"skipped {tgt_path_flac} because it already exists")
                                seconds += source[2]
                            else:
                                skips += 1
                    new_int_hours = int(seconds / 3600)
                    if new_int_hours != prev_int_hours:
                        prev_int_hours = new_int_hours
                        print(f"added {round(seconds / 3600, 2)} hours of audio")
                    previous_source = current_source
            print(f"added {round(seconds / 3600, 2)} hours of audio")
            print(f"skipped {skips} files, that's {round(skips / i * 100, 2)}%")
        super().__init__(**kwargs)

    def collect_data(self, directory):
        num_data = 0
        for file in Path(directory).glob("**/*.flac"):
            if file.name.startswith("."):
                continue
            num_data += 1
            yield {
                "path": file,
                "speaker": file.parent.name,
                "transcript": open(file.with_suffix(".lab"), "r").read(),
            }
        print(f"found {num_data} data points")
