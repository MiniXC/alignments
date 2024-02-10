# simply aligns a single or multiple audio files to a single or multiple text files

import random
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, List
import shutil

from rich.console import Console

console = Console()

from alignments.datasets.abstract import AbstractDataset


class AbstractAligner(ABC):
    """
    Abstract class for aligning audio files to text files.

    This class is meant to be subclassed by aligners that align audio files to text files.
    """

    ALLOWED_AUDIO_EXTENSIONS = [
        ".mp3",
        ".wav",
        ".aac",
        ".ogg",
        ".flac",
        ".avr",
        ".cdda",
        ".cvs",
        ".vms",
        ".aiff",
        ".au",
        ".amr",
        ".mp2",
        ".mp4",
        ".ac3",
        ".avi",
        ".wmv",
        ".mpeg",
        ".ircam",
        ".ark",
        ".scp",
    ]
    # for now, only .txt and .lab files are supported, but in the future, we may want to support more file types (e.g. .srt, .vtt, .TextGrid)
    ALLOWED_TEXT_EXTENSIONS = [".txt", ".lab"]

    def __init__(
        self,
        train_dataset: AbstractDataset = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the aligner
        :param train_aligner_with_subset: whether to train the aligner with a subset of the data
        :param subset_size: size of the subset to use for training
        :param kwargs: additional arguments
        """
        self.data_dict = {}
        # the supported audio and text file extensions can be overridden by subclasses
        super().__init__(**kwargs)

    @abstractmethod
    def _train(self, audio_paths: List[str], text_paths: List[str]) -> None:
        """
        Trains the aligner
        :param audio_paths: list of paths to audio files
        :param text_paths: list of paths to text files
        """
        pass

    @abstractmethod
    def _align(
        self, audio_paths: List[str], text_paths: List[str], alignment_dir: str
    ) -> List[str]:
        """
        Aligns audio files to text files
        :param audio_paths: list of paths to audio files
        :param text_paths: list of paths to text files
        :param alignment_dir: directory to save the alignment files
        :return: list of paths to output files
        """
        pass

    def train(self) -> None:
        """
        Trains the aligner on all of the data
        """
        audio_paths = [str(path) for path, _ in self.data_dict.values()]
        text_paths = [str(path) for _, path in self.data_dict.values()]
        self._train(audio_paths, text_paths)

    def align(
        self,
        audio_paths: List[str],
        text_paths: List[str],
        alignment_dir: str,
        overwrite: bool = False,
    ) -> None:
        """
        Aligns audio files to text files
        :param audio_paths: list of paths to audio files
        :param text_paths: list of paths to text files
        :param alignment_dir: directory to save the alignment files
        :param overwrite: whether to overwrite existing alignment files
        """
        output_paths = self._align(audio_paths, text_paths, alignment_dir)
        for output_path, text_path in zip(output_paths, text_paths):
            output_path = Path(output_path)
            if overwrite and output_path.exists():
                shutil.rmtree(output_path)
            if not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(text_path, "r") as f:
                    text = f.read()
                with open(output_path, "w") as f:
                    f.write(text)
                print(f"Saved alignment to {output_path}")

    def __call__(self, alignment_dir: str, overwrite: bool = False) -> None:
        """
        Aligns audio files to text files
        :param alignment_dir: directory to save the alignment files
        :param overwrite: whether to overwrite existing alignment files
        """
        audio_paths = [str(path) for path, _ in self.data_dict.values()]
        text_paths = [str(path) for _, path in self.data_dict.values()]
        self.align(audio_paths, text_paths, alignment_dir, overwrite)
