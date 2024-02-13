import tempfile
from pathlib import Path
import os

from rich.console import Console
from matplotlib import pyplot as plt
from tqdm import tqdm
import requests

from alignments.aligners.mfa import MFAAligner
from alignments.datasets.directory_dataset import DirectoryDataset

console = Console()

TEMP_DIR = tempfile.gettempdir()

dataset, aligner, aligner_g2p, example_alignment = None, None, None, None


def test_load_directory_dataset():
    """
    Test loading a directory dataset
    """
    global dataset
    # download dev-clean dataset from https://www.openslr.org/12 for tests to temp directory, if not already downloaded
    dataset_dir = Path(TEMP_DIR) / "LibriSpeech/dev-clean"
    dataset_processed_dir = Path(TEMP_DIR) / "LibriSpeech/dev-clean-processed"
    console.rule("Checking for LibriSpeech dataset...")
    if not dataset_dir.exists():
        with console.status("Downloading LibriSpeech dataset..."):
            url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
            r = requests.get(url)
            with open(f"{TEMP_DIR}/dev-clean.tar.gz", "wb") as f:
                f.write(r.content)
            dataset_dir.mkdir()
        with console.status("Extracting LibriSpeech dataset..."):
            os.system(f"tar -xvf {TEMP_DIR}/dev-clean.tar.gz -C {TEMP_DIR}")
    console.rule("Checking for processed LibriSpeech dataset...")
    if not dataset_processed_dir.exists():
        with console.status("Processing LibriSpeech dataset..."):
            # create processed dataset by copying the original dataset audio files and creating text files from .trans.txt
            dataset_processed_dir.mkdir()
            for speaker_dir in dataset_dir.iterdir():
                if speaker_dir.is_dir():
                    print(speaker_dir)
                    transcript_files = speaker_dir.rglob("*.trans.txt")
                    transcript_files = list(transcript_files)
                    if len(transcript_files) == 0:
                        continue
                    transcript_dict = {}
                    for transcript_file in transcript_files:
                        with open(transcript_file, "r") as f:
                            for line in f:
                                line = line.strip()
                                audio_file, transcript = line.split(" ", 1)
                                audio_file = transcript_file.parent / (
                                    audio_file + ".flac"
                                )
                                transcript_dict[audio_file] = (
                                    speaker_dir.name,
                                    transcript,
                                )
                    for audio_file, transcript in transcript_dict.items():
                        speaker_dir = dataset_processed_dir / transcript[0]
                        speaker_dir.mkdir(exist_ok=True)
                        with open(speaker_dir / f"{audio_file.stem}.lab", "w") as f:
                            f.write(transcript[1].lower())
                        os.symlink(audio_file, speaker_dir / audio_file.name)

    # create dataset
    dataset = DirectoryDataset(dataset_processed_dir)


def test_create_mfa_aligner():
    """
    Test creating an MFA aligner
    """
    global aligner
    aligner = MFAAligner()


def test_create_mfa_g2p_aligner():
    """
    Test creating an MFA aligner with a g2p model
    """
    global aligner_g2p
    aligner_g2p = MFAAligner(mfa_g2p_model="english_us_mfa")
    aligner_g2p.g2p.word_list = ["this", "is", "a", "test"]
    assert aligner_g2p.g2p.generate_pronunciations() == {
        "this": ["θ ɪ s", "θ ɪ z"],
        "is": ["ɪ s", "ɪ z", "aj s"],
        "a": ["ə"],
        "test": ["tʰ ɛ s t"],
    }


def test_align_single_mfa_aligner():
    """
    Test aligning with an MFA aligner
    """
    global example_alignment

    example_alignment = aligner.align_one(
        dataset.get_audio_text_pairs()[0][0],
        dataset.get_audio_text_pairs()[0][1],
    )

    assert example_alignment.word_segments[1].label == "mister"
    assert example_alignment.word_segments[1].start == 0.52
    assert example_alignment.word_segments[1].end == 0.79
    assert example_alignment.word_segments[-1].label == "<eps>"
    assert example_alignment.word_segments[-1].start == 5.44
    assert example_alignment.word_segments[-1].end == 5.855
    assert example_alignment.phone_segments[1].label == "mʲ"
    assert example_alignment.phone_segments[1].start == 0.52
    assert example_alignment.phone_segments[1].end == 0.59
    assert example_alignment.phone_segments[-1].label == "sil"
    assert example_alignment.phone_segments[-1].start == 5.44
    assert example_alignment.phone_segments[-1].end == 5.855


def test_align_single_mfa_g2p_aligner():
    """
    Test aligning with an MFA aligner with a g2p model
    """
    alignment = aligner_g2p.align_one(
        dataset.get_audio_text_pairs()[1][0],
        dataset.get_audio_text_pairs()[1][1],
    )

    assert alignment.word_segments[4].label == "quilter's"
    assert alignment.word_segments[4].start == 1.36
    assert alignment.word_segments[4].end == 1.83
    assert alignment.phone_segments[11].label == "cʷ"
    assert alignment.phone_segments[11].start == 1.36
    assert alignment.phone_segments[11].end == 1.47


def test_plot_alignment():
    """
    Test plotting an alignment
    """

    fig = example_alignment.plot()
    assert fig is not None
    plt.savefig(Path(TEMP_DIR) / "alignment.png")
    print(f"Alignment plot saved to {TEMP_DIR}/alignment.png")


def test_align_more():
    """
    Test aligning the first 10 files in the dataset
    """
    i = 0
    for audio_path, text_path in tqdm(dataset.get_audio_text_pairs(), desc="Aligning"):
        aligner_g2p.align_one(audio_path, text_path)
        i += 1
        if i == 10:
            break
    assert i == 10
