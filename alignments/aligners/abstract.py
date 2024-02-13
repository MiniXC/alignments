"""
abstract classes for aligners
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, List
import json

from rich.console import Console
from matplotlib import pyplot as plt
import matplotlib as mpl
import torchaudio

from alignments.datasets.abstract import AbstractDataset


console = Console()


class Interval:
    """
    Class for representing an interval.
    An interval consists of a start and end time and a label.
    """

    def __init__(self, start: float, end: float, label: str) -> None:
        """
        Initializes the interval
        :param start: start time
        :param end: end time
        :param label: label
        """
        self.start = start
        self.end = end
        self.label = label

    def __repr__(self) -> str:
        return f"Interval({self.start}, {self.end}, {self.label})"

    def __str__(self) -> str:
        return f"({self.start}, {self.end}): {self.label}"


class Alignment:
    """
    Class for representing an alignment.
    An alignment consists of an audio file, a transcription, and the alignment between the two.
    The alignment can be represented as lists of word and phone segments.
    """

    def __init__(
        self,
        audio_path: Path,
        text_path: Path,
        word_segments: List[Interval],
        phone_segments: List[Interval],
    ) -> None:
        """
        Initializes the alignment
        :param audio_path: path to audio file
        :param text_path: path to text file
        :param word_segments: list of word segments
        :param phone_segments: list of phone segments
        """
        self.audio_path = audio_path
        self.text_path = text_path
        self.word_segments = word_segments
        self.phone_segments = phone_segments

    def __repr__(self) -> str:
        return f"Alignment({self.audio_path}, {self.text_path}, {self.word_segments}, {self.phone_segments})"

    def __str__(self) -> str:
        return_str = f"Alignment for {self.audio_path} and {self.text_path}:\n"
        return_str += "Word segments:\n"
        for segment in self.word_segments:
            return_str += f"{segment}\n"
        return_str += "Phone segments:\n"
        for segment in self.phone_segments:
            return_str += f"{segment}\n"
        return return_str

    def plot(
        self,
        plot_words: bool = True,
        plot_phones: bool = True,
        **kwargs: Any,
    ) -> mpl.figure.Figure:
        """
        Plots the alignment
        :param plot_words: whether to plot word segments
        :param plot_phones: whether to plot phone segments
        :return: figure
        """
        waveform, sample_rate = torchaudio.load(self.audio_path)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate, n_mels=80, hop_length=256
        )(waveform)[0]
        if "figsize" not in kwargs:
            # set figsize to be proportional to the length of the audio
            len_in_seconds = mel.shape[1] / sample_rate * 256 * 4
            kwargs["figsize"] = (len_in_seconds, 7.5)
        fig, ax = plt.subplots(**kwargs)
        ax.imshow(
            mel.log2().detach().numpy(),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="bilinear",
        )
        top_offset = plt.ylim()[1]
        if plot_words:
            for segment in self.word_segments:
                ax.text(
                    (segment.start + segment.end) / 2 * sample_rate / 256,
                    top_offset,
                    segment.label,
                    ha="center",
                    va="bottom",
                    color="r",
                )
                # line between labels
                ax.plot(
                    [
                        segment.start * sample_rate / 256,
                        segment.end * sample_rate / 256,
                    ],
                    [top_offset, top_offset],
                    color="r",
                )
                top_offset += 2 / len(self.word_segments)
        if plot_phones:
            for segment in self.phone_segments:
                ax.text(
                    (segment.start + segment.end) / 2 * sample_rate / 256,
                    top_offset,
                    segment.label,
                    ha="center",
                    va="bottom",
                    color="g",
                )
                # line between labels
                ax.plot(
                    [
                        segment.start * sample_rate / 256,
                        segment.end * sample_rate / 256,
                    ],
                    [top_offset, top_offset],
                    color="g",
                )
                top_offset += 2 / len(self.phone_segments)
        plt.xlim(0, mel.shape[1])
        # show x-axis in seconds
        ax.set_xtick_labels = [f"{i:.2f}" for i in ax.get_xticks() / sample_rate * 256]
        plt.tight_layout()
        ax.set_xlabel("Time (s)")
        return fig

    @classmethod
    def from_json(
        cls, json_path: Path, audio_path: Path, text_path: Path
    ) -> "Alignment":
        """
        Creates an alignment from a JSON file
        :param json_path: path to JSON file
        :param audio_path: path to audio file
        :param text_path: path to text file
        :return: alignment
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        words = data["tiers"]["words"]["entries"]
        phones = data["tiers"]["phones"]["entries"]
        word_segments = [Interval(start, end, label) for start, end, label in words]
        phone_segments = [Interval(start, end, label) for start, end, label in phones]
        return cls(audio_path, text_path, word_segments, phone_segments)


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
    # for now, only .lab files are supported, but in the future,
    # we may want to support more file types (e.g. .srt, .vtt, .TextGrid)
    ALLOWED_TEXT_EXTENSIONS = [".lab"]

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the aligner
        :param kwargs: additional arguments
        """
        self.data_dict = {}
        # the supported audio and text file extensions can be overridden by subclasses
        super().__init__(**kwargs)

    @abstractmethod
    def align_one(self, audio_path: Path, text_path: Path) -> Alignment:
        """
        Aligns audio files to text files
        :param audio_paths: list of paths to audio files
        :param text_paths: list of paths to text files
        :param alignment_dir: directory to save the alignment files
        :return: list of paths to output files
        """
        pass

    def align_dataset(
        self,
        dataset: AbstractDataset,
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
        for audio_path, text_path in dataset.get_audio_text_pairs():
            alignment_path = Path(alignment_dir) / f"{audio_path.stem}.lab"
            if alignment_path.exists() and not overwrite:
                console.print(
                    f"Alignment file {alignment_path} already exists. Skipping...",
                    style="warning",
                )
                continue
            alignment = self.align_one(audio_path, text_path)
            with open(alignment_path, "w", "utf-8") as f:
                for segment in alignment.phone_segments:
                    f.write(
                        f"{segment.start:.2f}\t{segment.end:.2f}\t{segment.label}\n"
                    )
            console.print(
                f"Alignment file {alignment_path} saved successfully.",
                style="info",
            )
