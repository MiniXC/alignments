from pathlib import Path
from typing import List


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
