from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import random

from rich.console import Console

from alignments.aligners.abstract import AbstractAligner
from alignments.datasets.exceptions import (
    DatasetStructureException,
    DuplicateAudioException,
    DuplicateTextException,
    NoAudioException,
    NoTextException,
    EmptyDatasetException,
)

from alignments.datasets.abstract import AbstractDataset

console = Console()


class DirectoryDataset(AbstractDataset):
    """
    Class for representing a dataset of audio paired with text, where the audio and text files are in a directory.
    """

    def __init__(self, directory: Optional[Union[Path, str]] = None) -> None:
        """
        Initializes the dataset
        """
        super().__init__()
        self.data_dict = {}
        if directory:
            self._load_data_from_path(directory)

    def get_audio_text_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Returns a list of audio and text file paths
        """
        return [pair[1:] for pair in sorted(self.data_dict.values())]

    def get_audio_text_pairs_by_speaker(self) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Returns a list of audio and text file paths, grouped by speaker
        """
        speaker_dict = {}
        for value in sorted(self.data_dict.values()):
            speaker = value[0]
            if speaker not in speaker_dict:
                speaker_dict[speaker] = []
            speaker_dict[speaker].append(value[1:])
        return speaker_dict

    def _load_data_from_path(self, path: Union[Path, str]) -> None:
        """
        Finds paired audio and text paths from a given path.
        Assumes the directory structure is as follows:
        path
        ├── speaker1
        │   ├── audio1.wav
        │   ├── text1.txt
        │   ├── ...
        └── speaker2
            ├── audio1.wav
            ├── text1.txt
            ├── ...
        If there is only one speaker, the directory structure can be as follows:
        path
        ├── audio1.wav
        ├── text1.txt
        ├── ...
        Any audio formats readable by torchaudio.load and any text file format readable by open() are supported.
        :param path: path to the directory containing the audio and text files
        """
        single_speaker = None
        warn_structure = False
        for speaker in (path).iterdir():
            if speaker.is_dir():
                if single_speaker:
                    raise DatasetStructureException(single_speaker, path)
                console.log(f"Loading data from {speaker}")
                for file in speaker.rglob("*"):
                    if file.is_file():
                        self._add_audio_and_text_paths(file, speaker.name)
                    elif file.is_dir():
                        warn_structure = True
                single_speaker = False
            else:
                if single_speaker is False:
                    raise DatasetStructureException(single_speaker, path)
                console.log(f"Loading data from single speaker")
                self._add_audio_and_text_paths(speaker, "1")
                single_speaker = True
        if not self.data_dict:
            raise EmptyDatasetException(path)
        if warn_structure:
            console.log(
                "[yellow]Warning: Some files were found in subdirectories more than one level deep. Only top level directories will be used as speaker directories.[/yellow]"
            )

    def _add_audio_and_text_paths(self, path: Path, speaker: str) -> None:
        """
        Adds audio and text paths to the data dictionary
        :param path: path to the audio or text file
        """
        if path.suffix in AbstractAligner.ALLOWED_AUDIO_EXTENSIONS:
            audio_path = path
            key = audio_path.stem
            if key in self.data_dict and self.data_dict[key][1] != audio_path:
                raise DuplicateAudioException([path, self.data_dict[key][0]])
            return_candidate = None
            for ext in AbstractAligner.ALLOWED_TEXT_EXTENSIONS:
                text_path = path.with_suffix(ext)
                if text_path.exists():
                    if return_candidate is not None and return_candidate != text_path:
                        raise DuplicateTextException([path, return_candidate])
                    return_candidate = text_path
            if return_candidate is None:
                raise NoTextException(f"No text file found for {path}")
            self.data_dict[key] = (speaker, audio_path, return_candidate)
        elif path.suffix in AbstractAligner.ALLOWED_TEXT_EXTENSIONS:
            text_path = path
            key = text_path.stem
            if key in self.data_dict and self.data_dict[key][2] != text_path:
                raise DuplicateTextException([path, self.data_dict[key][1]])
            return_candidate = None
            for ext in AbstractAligner.ALLOWED_AUDIO_EXTENSIONS:
                audio_path = path.with_suffix(ext)
                if audio_path.exists():
                    if return_candidate is not None and return_candidate != audio_path:
                        raise DuplicateAudioException([path, return_candidate])
                    return_candidate = audio_path
            if return_candidate is None:
                raise NoAudioException(path)
            self.data_dict[key] = (speaker, return_candidate, text_path)

    def get_subset(self, length: int, seed: int = 42) -> "DirectoryDataset":
        """
        Returns a random subset of the dataset
        """
        random.seed(seed)
        subset = random.sample(list(self.data_dict.values()), length)
        new_dataset = DirectoryDataset()
        new_dataset.data_dict = {str(i): pair for i, pair in enumerate(subset)}
        return new_dataset

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        return len(self.data_dict)
