from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import tempfile

from kalpy.utterance import Segment, Utterance
from kalpy.fstext.lexicon import LexiconCompiler, HierarchicalCtm
from kalpy.feat.cmvn import CmvnComputer
import montreal_forced_aligner
from montreal_forced_aligner.models import ModelManager
from montreal_forced_aligner.g2p.generator import PyniniGenerator
from montreal_forced_aligner.models import MODEL_TYPES, AcousticModel
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.online.alignment import align_utterance_online
from rich.console import Console

console = Console()

from alignments.aligners.abstract import AbstractAligner
from alignments.aligners.exceptions import (
    MFAVersionException,
    MFANoKaldiException,
    MFAMissingPronunciationException,
    MFADictionaryFormatException,
    MFAMultipleUttException,
)


class PyniniLocalGenerator(PyniniGenerator):

    @property
    def working_directory(self) -> Path:
        """
        create a temporary directory for the working directory
        """
        if self._working_directory is None:
            self._working_directory = Path(tempfile.mkdtemp())
        return self._working_directory


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
        mfa_dictionary: str = "english_mfa",
        mfa_g2p_model: Optional[str] = None,
        mfa_acoustic_model_ignore_cache: bool = False,
        mfa_beam: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the aligner
        """
        super().__init__(**kwargs)

        self.beam = mfa_beam

        try:
            import _kalpy
        except ImportError:
            raise MFANoKaldiException()

        mfa_version = montreal_forced_aligner.utils.get_mfa_version()
        if not mfa_version.startswith("3."):
            raise MFAVersionException(mfa_version)

        manager = ModelManager(ignore_cache=mfa_acoustic_model_ignore_cache)

        console.rule("Checking for MFA acoustic model [italic]english_mfa[/italic]...")

        acoustic_model = MODEL_TYPES["acoustic"].get_pretrained_path(mfa_acoustic_model)
        if acoustic_model is None:
            manager.download_model("acoustic", mfa_acoustic_model)
            acoustic_model = manager.get_pretrained_path("acoustic", mfa_acoustic_model)
        self.acoustic_model = AcousticModel(acoustic_model)

        console.rule("Checking for MFA dictionary [italic]english_mfa[/italic]...")
        dictionary = MODEL_TYPES["dictionary"].get_pretrained_path(mfa_dictionary)
        if dictionary is None:
            manager.download_model("dictionary", mfa_dictionary)
            dictionary = manager.get_pretrained_path("dictionary", mfa_dictionary)
        if dictionary.suffix != ".dict":
            raise MFADictionaryFormatException(dictionary)
        self.lexicon = LexiconCompiler(
            disambiguation=False,
            silence_probability=self.acoustic_model.parameters["silence_probability"],
            initial_silence_probability=self.acoustic_model.parameters[
                "initial_silence_probability"
            ],
            final_silence_correction=self.acoustic_model.parameters[
                "final_silence_correction"
            ],
            final_non_silence_correction=self.acoustic_model.parameters[
                "final_non_silence_correction"
            ],
            silence_phone=self.acoustic_model.parameters["optional_silence_phone"],
            oov_phone=self.acoustic_model.parameters["oov_phone"],
            position_dependent_phones=self.acoustic_model.parameters[
                "position_dependent_phones"
            ],
            phones=self.acoustic_model.parameters["non_silence_phones"],
            ignore_case=True,
        )
        self.lexicon.load_pronunciations(dictionary)
        temp_lexicon_path = Path(tempfile.mkstemp()[1])
        l_fst_path = temp_lexicon_path.with_suffix(".fst")
        l_align_fst_path = temp_lexicon_path.with_suffix(".align.fst")
        words_path = temp_lexicon_path.with_suffix(".words.txt")
        phones_path = temp_lexicon_path.with_suffix(".phones.txt")
        self.lexicon.fst.write(str(l_fst_path))
        self.lexicon.align_fst.write(str(l_align_fst_path))
        self.lexicon.word_table.write_text(words_path)
        self.lexicon.phone_table.write_text(phones_path)
        self.lexicon.clear()

        self.cmvn = CmvnComputer()

        if mfa_g2p_model is not None:
            console.rule("Checking for MFA g2p model [italic]g2p[/italic]...")
            model_path = MODEL_TYPES["g2p"].get_pretrained_path(mfa_g2p_model)
            if model_path is None:
                manager.download_model("g2p", mfa_g2p_model)
                model_path = manager.get_pretrained_path("g2p", mfa_g2p_model)
            self.g2p = PyniniGenerator(None, model_path)
            self.g2p.setup()

    def _align(self, audio_path: Path, text_path: Path, output_dir: Path) -> Path:
        """
        Aligns an audio file to a text file
        """
        file_name = audio_path.stem
        file = FileData.parse_file(file_name, audio_path, text_path, "", 0)
        file_ctm = HierarchicalCtm([])
        utts = file.utterances
        if len(utts) > 1:
            raise MFAMultipleUttException(text_path)
        utt = utts[0]
        seg = Segment(audio_path, utt.begin, utt.end, utt.channel)
        utt = Utterance(seg, utt.text)
        utt.generate_mfccs(self.acoustic_model.mfcc_computer)
        cmvn = self.cmvn.compute_cmvn_from_features([utt.mfccs])
        utt.apply_cmvn(cmvn)
        ctm = align_utterance_online(
            self.acoustic_model,
            utt,
            self.lexicon,
            beam=self.beam,
        )
        file_ctm.word_intervals.extend(ctm.word_intervals)
        output_path = output_dir / f"{file_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_ctm.export_textgrid(
            output_path,
            file_duration=file.wav_info.duration,
            output_format="json",
        )
        return output_path

    def _train(
        self, audio_paths: List[Path], text_paths: List[Path], output_dir: Path
    ) -> None:
        """
        Trains the aligner on a set of audio files and text files
        """
        pass
