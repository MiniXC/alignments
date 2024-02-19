from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import tempfile

from kalpy.utterance import Segment, Utterance
from kalpy.fstext.lexicon import LexiconCompiler, HierarchicalCtm, Pronunciation
from kalpy.feat.cmvn import CmvnComputer
from kalpy.decoder.training_graphs import TrainingGraphCompiler
from kalpy.gmm.align import GmmAligner
import montreal_forced_aligner
from montreal_forced_aligner.models import ModelManager
from montreal_forced_aligner.g2p.generator import PyniniGenerator
from montreal_forced_aligner.models import MODEL_TYPES, AcousticModel
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.online.alignment import align_utterance_online
from montreal_forced_aligner import config as mfa_config
from montreal_forced_aligner.data import (
    BRACKETED_WORD,
    CUTOFF_WORD,
    LAUGHTER_WORD,
    OOV_WORD,
    Language,
)
from montreal_forced_aligner.dictionary.mixins import (
    DEFAULT_BRACKETS,
    DEFAULT_CLITIC_MARKERS,
    DEFAULT_COMPOUND_MARKERS,
    DEFAULT_PUNCTUATION,
    DEFAULT_WORD_BREAK_MARKERS,
)
from montreal_forced_aligner.tokenization.simple import SimpleTokenizer
from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer
from rich.console import Console

from alignments.aligners.abstract import AbstractAligner, Alignment
from alignments.aligners.exceptions import (
    MFAVersionException,
    MFANoKaldiException,
    MFAMissingPronunciationException,
    MFADictionaryFormatException,
    MFAMultipleUttException,
    MFAAlignmentException,
)

console = Console()


ALLOWED_AUDIO_EXTENSIONS = [
    ".wav",
]
ALLOWED_TEXT_EXTENSIONS = [".lab"]

mfa_config.GLOBAL_CONFIG.current_profile.update({"use_mp": False, "always_quiet": True})
mfa_config.GLOBAL_CONFIG.save()
mfa_config.QUIET = True
mfa_config.USE_MP = False


# this is a dirty hack to get around the fact that the lexicon is not picklable
class FakeLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MFAligner(AbstractAligner):
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
        mfa_retry_beam: int = 40,
        mfa_transition_scale: float = 1.0,
        mfa_acoustic_scale: float = 0.1,
        mfa_self_loop_scale: float = 0.1,
        mfa_boost_silence: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the aligner
        """
        super().__init__(**kwargs)

        try:
            import _kalpy
        except ImportError as e:
            raise MFANoKaldiException() from e

        mfa_version = montreal_forced_aligner.utils.get_mfa_version()
        if not mfa_version.startswith("3."):
            raise MFAVersionException(mfa_version)

        manager = ModelManager(ignore_cache=mfa_acoustic_model_ignore_cache)

        console.rule("Checking for MFA acoustic model [italic]english_mfa[/italic]...")

        acoustic_model = MODEL_TYPES["acoustic"].get_pretrained_path(mfa_acoustic_model)
        if acoustic_model is None:
            manager.download_model("acoustic", mfa_acoustic_model)
            acoustic_model = MODEL_TYPES["acoustic"].get_pretrained_path(
                mfa_acoustic_model
            )
        self.acoustic_model = AcousticModel(acoustic_model)

        console.rule("Checking for MFA dictionary [italic]english_mfa[/italic]...")
        dictionary = MODEL_TYPES["dictionary"].get_pretrained_path(mfa_dictionary)
        if dictionary is None:
            manager.download_model("dictionary", mfa_dictionary)
            dictionary = MODEL_TYPES["dictionary"].get_pretrained_path(mfa_dictionary)
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

        self.oov_cache = {}

        # this is a dirty hack to get around the fact that the lexicon is not picklable
        self.lexicon.lock = FakeLock()

        self.cmvn = CmvnComputer()

        self.gmm_aligner = GmmAligner(
            self.acoustic_model.alignment_model_path,
            beam=mfa_beam,
            retry_beam=mfa_retry_beam,
            transition_scale=mfa_transition_scale,
            acoustic_scale=mfa_acoustic_scale,
            self_loop_scale=mfa_self_loop_scale,
        )
        if mfa_boost_silence != 1.0:
            self.gmm_aligner.boost_silence(
                mfa_boost_silence, self.lexicon.silence_symbols
            )
        self.gmm_aligner_kwargs = {
            "beam": mfa_beam,
            "retry_beam": mfa_retry_beam,
            "transition_scale": mfa_transition_scale,
            "acoustic_scale": mfa_acoustic_scale,
            "self_loop_scale": mfa_self_loop_scale,
            "boost_silence": mfa_boost_silence,
        }

        self.tokenizer = SimpleTokenizer(
            word_table=self.lexicon.word_table,
            word_break_markers=DEFAULT_WORD_BREAK_MARKERS,
            punctuation=DEFAULT_PUNCTUATION,
            clitic_markers=DEFAULT_CLITIC_MARKERS,
            compound_markers=DEFAULT_COMPOUND_MARKERS,
            brackets=DEFAULT_BRACKETS,
            laughter_word=LAUGHTER_WORD,
            oov_word=OOV_WORD,
            bracketed_word=BRACKETED_WORD,
            cutoff_word=CUTOFF_WORD,
            ignore_case=True,
        )

        if mfa_g2p_model is not None:
            console.rule("Checking for MFA g2p model [italic]g2p[/italic]...")
            model_path = MODEL_TYPES["g2p"].get_pretrained_path(mfa_g2p_model)
            if model_path is None:
                manager.download_model("g2p", mfa_g2p_model)
                model_path = MODEL_TYPES["g2p"].get_pretrained_path(mfa_g2p_model)
            self.g2p = PyniniGenerator(None, model_path)
            self.g2p.setup()

    def _align_utterance_online_faster(
        self,
        utterance: Utterance,
    ) -> HierarchicalCtm:
        text = utterance.transcript

        if utterance.mfccs is None:
            utterance.generate_mfccs(self.acoustic_model.mfcc_computer)
            if self.acoustic_model.uses_cmvn:
                if cmvn is None:
                    cmvn_computer = CmvnComputer()
                    cmvn = cmvn_computer.compute_cmvn_from_features([utterance.mfccs])
                utterance.apply_cmvn(cmvn)
        feats = utterance.generate_features(
            self.acoustic_model.mfcc_computer,
            self.acoustic_model.pitch_computer,
            lda_mat=self.acoustic_model.lda_mat,
            fmllr_trans=None,
        )

        graph_compiler = TrainingGraphCompiler(
            self.acoustic_model.alignment_model_path,
            self.acoustic_model.tree_path,
            self.lexicon,
            self.lexicon.word_table,
        )

        fst = graph_compiler.compile_fst(text)
        alignment = self.gmm_aligner.align_utterance(fst, feats)
        if alignment is None:
            raise MFAAlignmentException(
                f"Could not align the file with the current beam size ({self.gmm_aligner.beam}, "
                "please try increasing the beam size via `--beam X`"
            )
        phone_intervals = alignment.generate_ctm(
            self.gmm_aligner.transition_model,
            self.lexicon.phone_table,
            self.acoustic_model.mfcc_computer.frame_shift,
        )
        ctm = self.lexicon.phones_to_pronunciations(
            utterance.transcript, alignment.words, phone_intervals, transcription=False
        )
        ctm.likelihood = alignment.likelihood
        ctm.update_utterance_boundaries(utterance.segment.begin, utterance.segment.end)
        return ctm

    def align_one(
        self, audio_path: Path, text_path: Path, fast: bool = True
    ) -> Alignment:
        """
        Aligns an audio file to a text file.

        :param audio_path: path to the audio file
        :param text_path: path to the text file
        :param fast: whether to use the faster alignment method
        :return: the alignment
        """
        file_name = audio_path.stem
        file = FileData.parse_file(file_name, audio_path, text_path, "", 0)
        file_ctm = HierarchicalCtm([])
        utts = file.utterances
        if len(utts) > 1:
            raise MFAMultipleUttException(text_path)
        utt = utts[0]
        utt.text = utt.text.lower()
        seg = Segment(audio_path, utt.begin, utt.end, utt.channel)
        unk_words = []
        text, _, oovs = self.tokenizer(utt.text)
        for word in oovs:
            if not self.lexicon.word_table.member(word):
                if hasattr(self, "g2p"):
                    if word not in unk_words:
                        unk_words.append(word)
                    continue
                raise MFAMissingPronunciationException(word)
        if unk_words:
            self.g2p.word_list = unk_words
            g2p_prons = self.g2p.generate_pronunciations()
            for word, prons in g2p_prons.items():
                for pron in prons:
                    self.lexicon.add_pronunciation(
                        Pronunciation(word, pron, None, None, None, None, None)
                    )
        utt = Utterance(seg, text)
        utt.generate_mfccs(self.acoustic_model.mfcc_computer)
        cmvn = self.cmvn.compute_cmvn_from_features([utt.mfccs])
        utt.apply_cmvn(cmvn)
        if fast:
            ctm = self._align_utterance_online_faster(utt)
        else:
            ctm = align_utterance_online(
                self.acoustic_model,
                utt,
                self.lexicon,
                **self.gmm_aligner_kwargs,
            )
        file_ctm.word_intervals.extend(ctm.word_intervals)
        temp_output_dir = Path(tempfile.mkdtemp())
        output_path = temp_output_dir / f"{file_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_ctm.export_textgrid(
            output_path,
            file_duration=file.wav_info.duration,
            output_format="json",
        )
        alignment = Alignment.from_json(output_path, audio_path, text_path)
        return alignment
