from glob import glob
from pathlib import Path
import re

from alignments.dataset import AlignmentDataset

class LibrittsDataset(AlignmentDataset):
    def __init__(
        self,
        target_directory,
        source_directory,
        source_url=None,
        force="none",
        symbolic_links=True,
        verbose=False,
        show_warnings=False,
        punctuation_marks="!?.,;",
    ):
        super().__init__(
            target_directory,
            source_directory,
            source_url,
            force,
            symbolic_links,
            acoustic_model="english_us_arpa",
            g2p_model="english_us_arpa",
            lexicon="http://www.openslr.org/resources/11/librispeech-lexicon.txt",
            verbose=verbose,
            show_warnings=show_warnings,
            punctuation_marks=punctuation_marks,
        )

    def collect_data(self, directory):
        for file in Path(directory).glob("**/*.wav"):
            transcript = open(file.with_suffix(".normalized.txt"), "r").read()
            if "illustration" in transcript.lower():
                transcript = transcript.replace("Illustration:", " ")
            transcript = re.sub("[\[\]\(\)-]", " ", transcript)
            transcript = re.sub("\s+", " ", transcript)
            yield {
                "path": file,
                "speaker": file.parent.parent.name,
                "transcript": transcript.upper(),
            }