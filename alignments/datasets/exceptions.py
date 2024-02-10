from pathlib import Path
from typing import List


class DatasetStructureException(Exception):
    """
    Exception raised when the dataset structure is not as expected
    """

    def __init__(self, single_speaker: bool, path: Path):
        message = (
            f"Multiple speakers found in the dataset when only one was expected."
            if single_speaker is False
            else "Individual audio and text files found in the dataset when speaker directories were expected."
        )
        message += f" Please check the structure of the dataset at {path}"
        super().__init__(message)


class DuplicateAudioException(Exception):
    """
    Exception raised when multiple audio files are found for a single file
    """

    def __init__(self, paths: List[Path]):
        super().__init__(
            f"Multiple audio files with different extensions found for {paths[0].stem}: {paths[0].suffix} and {paths[1].suffix}"
        )
        self.paths = paths


class DuplicateTextException(Exception):
    """
    Exception raised when multiple text files are found for a single file
    """

    def __init__(self, paths: List[Path]):
        super().__init__(
            f"Multiple text files with different extensions found for {paths[0].stem}: {paths[0].suffix} and {paths[1].suffix}"
        )
        self.paths = paths


class NoAudioException(Exception):
    """
    Exception raised when no audio file is found for a text file
    """

    def __init__(self, path: Path):
        super().__init__(f"No audio file found for {path}")
        self.path = path


class NoTextException(Exception):
    """
    Exception raised when no text file is found for an audio file
    """

    def __init__(self, path: Path):
        super().__init__(f"No text file found for {path}")
        self.path = path


class EmptyDatasetException(Exception):
    """
    Exception raised when the dataset is empty
    """

    def __init__(self, path: str):
        super().__init__(f"No audio and text files found in {path}")
        self.path = path
