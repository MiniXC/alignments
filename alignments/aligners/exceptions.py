from pathlib import Path


class MFAVersionException(Exception):
    """
    Exception raised when the MFA version is not supported
    """

    def __init__(self, version: str):
        super().__init__(
            f"MFA version {version} is not supported, please use version 3.x"
        )
        self.version = version


class MFANoKaldiException(Exception):
    """
    Exception raised when Kaldi is not found
    """

    def __init__(self):
        super().__init__(
            "_kalpy package (Kaldi C bindings) not found, please install kalpy with Kaldi bindings - See https://pypi.org/project/kalpy-kaldi/ for more information."
        )


class MFAMissingPronunciationException(Exception):
    """
    Exception raised when a word is not found in the pronunciation dictionary (and no g2p model is provided)
    """

    def __init__(self, word: str):
        super().__init__(
            f"Word '{word}' not found in the pronunciation dictionary, please provide a g2p model"
        )
        self.word = word


class MFADictionaryFormatException(Exception):
    """
    Exception raised when the pronunciation dictionary is not in the expected format
    """

    def __init__(self, path: Path):
        super().__init__(f"Dictionary file at {path} is not in the expected format")
        self.path = path


class MFAMultipleUttException(Exception):
    """
    Exception raised when multiple utterances are found in a single text file
    """

    def __init__(self, path: Path):
        super().__init__(
            f"Multiple utterances found in {path}, currently only one utterance per text file is supported"
        )
        self.path = path


class MFAAlignmentException(Exception):
    """
    Exception raised when an error occurs during alignment
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
