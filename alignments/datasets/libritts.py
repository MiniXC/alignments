from glob import glob
from pathlib import Path
import re

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
