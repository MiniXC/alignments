from glob import glob
from pathlib import Path
import re
from transformers.utils.hub import cached_file

from alignments.dataset import AlignmentDataset

class LibrittsDataset(AlignmentDataset):
    def __init__(self, **kwargs):
        if "acoustic_model" not in kwargs:
            kwargs["acoustic_model"] = "english_us_arpa"
        if "g2p_model" not in kwargs:
            kwargs["g2p_model"] = "english_us_arpa"
        if "lexicon" not in kwargs:
            kwargs["lexicon"] = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
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
        if "textgrid_url" not in kwargs and "source_url" in kwargs:
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
