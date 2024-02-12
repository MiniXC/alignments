import requests
import tempfile
from pathlib import Path
import os

from rich.console import Console

console = Console()

from alignments.aligners.mfa import MFAAligner
from alignments.datasets.directory_dataset import DirectoryDataset

TEMP_DIR = tempfile.gettempdir()


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
    align_dir = Path(TEMP_DIR) / "alignments"
    print(dataset.get_audio_text_pairs()[0])
    from time import time

    start = time()
    aligned_json = aligner._align(
        dataset.get_audio_text_pairs()[0][0],
        dataset.get_audio_text_pairs()[0][1],
        align_dir,
    )
    print(time() - start)
    expected_aligned_json = Path("tests/test_alignment.json")
    assert aligned_json.exists()
    assert aligned_json.read_text() == expected_aligned_json.read_text()


# def test_align_mfa_aligner():
#     """
#     Test aligning with an MFA aligner
#     """
#     aligner = MFAAligner()
#     align_dir = Path(TEMP_DIR) / "alignments"
#     aligner.align(dataset, align_dir, overwrite=True)


# def test_align_one_mfa_aligner():
#     """
#     Test aligning with an MFA aligner
#     """
#     aligner = MFAAligner()
#     align_dir = Path(TEMP_DIR) / "alignments"
#     alignment = aligner.align(dataset.get_audio_text_pairs()[0])
