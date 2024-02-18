from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, Dict
from pathlib import Path

import numpy as np


class AbstractDataset(ABC):
    """
    Abstract class for representing a dataset of audio paired with text.

    This class is meant to be subclassed by classes that represent datasets of audio and text files.
    """

    def __init__(self) -> None:
        """
        Initializes the dataset
        """
        pass

    @abstractmethod
    def get_audio_text_pairs(
        self,
    ) -> List[Union[Tuple[Path, Path], Tuple[np.ndarray, str]]]:
        """
        Returns a list of audio and text file paths or numpy arrays and strings
        """
        pass

    @abstractmethod
    def get_audio_text_pairs_by_speaker(
        self,
    ) -> Dict[str, List[Union[Tuple[Path, Path], Tuple[np.ndarray, str]]]]:
        """
        Returns a list of audio and text file paths (or numpy arrays and strings), grouped by speaker
        """
        pass

    @abstractmethod
    def get_subset(self, length: int, seed: int = 42) -> "AbstractDataset":
        """
        Returns a random subset of the dataset
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        pass
