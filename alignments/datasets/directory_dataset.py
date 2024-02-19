from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
from tqdm.contrib.concurrent import process_map
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

def flatten(xss):
    return [x for xs in xss for x in xs]

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
        if isinstance(directory, str):
            directory = Path(directory)
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

    def _process_dir(self, speaker):
        if speaker.is_dir():
            paths = []
            for file in speaker.rglob("*"):
                if file.is_file():
                    paths.append(self._add_audio_and_text_paths(file, speaker.name))
            return paths
        else:
            console.log(f"Loading data from single speaker")
            return self._add_audio_and_text_paths(speaker, "1")

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
        dirs = list((path).iterdir())
        results = process_map(self._process_dir, dirs, chunksize=10)
        if isinstance(results[0], list):
            results = flatten(results)
        for r in results:
            self.data_dict[r[0]] = r[1:]
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
            return_candidate = None
            for ext in AbstractAligner.ALLOWED_TEXT_EXTENSIONS:
                text_path = path.with_suffix(ext)
                if text_path.exists():
                    if return_candidate is not None and return_candidate != text_path:
                        raise DuplicateTextException([path, return_candidate])
                    return_candidate = text_path
            if return_candidate is None:
                raise NoTextException(f"No text file found for {path}")
            return (key, speaker, audio_path, return_candidate)
        elif path.suffix in AbstractAligner.ALLOWED_TEXT_EXTENSIONS:
            text_path = path
            key = text_path.stem
            return_candidate = None
            for ext in AbstractAligner.ALLOWED_AUDIO_EXTENSIONS:
                audio_path = path.with_suffix(ext)
                if audio_path.exists():
                    if return_candidate is not None and return_candidate != audio_path:
                        raise DuplicateAudioException([path, return_candidate])
                    return_candidate = audio_path
            if return_candidate is None:
                raise NoAudioException(path)
            return (key, speaker, return_candidate, text_path)

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
