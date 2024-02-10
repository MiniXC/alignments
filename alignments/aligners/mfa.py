from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import montreal_forced_aligner
from montreal_forced_aligner.models import ModelManager
from rich.console import Console

console = Console()

from alignments.aligners.abstract import AbstractAligner
from alignments.aligners.exceptions import MFAVersionException, MFANoKaldiException

ALLOWED_AUDIO_EXTENSIONS = [
    ".wav",
]
ALLOWED_TEXT_EXTENSIONS = [".lab"]


class MFAAligner(AbstractAligner):
    """
    Class for aligning audio files to text files using Montreal Forced Aligner
    """

    def __init__(
        self,
        mfa_acoustic_model: str = "english_mfa",
        mfa_acoustic_model_ignore_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the aligner
        """
        super().__init__(**kwargs)

        try:
            import _kalpy
        except ImportError:
            raise MFANoKaldiException()

        mfa_version = montreal_forced_aligner.utils.get_mfa_version()
        if not mfa_version.startswith("3."):
            raise MFAVersionException(mfa_version)

        console.rule("Checking for MFA acoustic model...")
        manager = ModelManager(ignore_cache=mfa_acoustic_model_ignore_cache)
        manager.download_model("acoustic", mfa_acoustic_model)
        self.acoustic_model = mfa_acoustic_model

        

    def _align(
        self, audio_path: Path, text_path: Path, output_dir: Path
    ) -> Tuple[Path, Path]:
        """
        Aligns an audio file to a text file
        """
        pass

    def _train(
        self, audio_paths: List[Path], text_paths: List[Path], output_dir: Path
    ) -> None:
        """
        Trains the aligner on a set of audio files and text files
        """
        pass
