import requests
import tempfile
from pathlib import Path
import os

from rich.console import Console

console = Console()

from alignments.aligners.mfa import MFAAligner
from alignments.datasets.directory_dataset import DirectoryDataset

TEMP_DIR = tempfile.gettempdir()


# download dev-clean dataset from https://www.openslr.org/12 for tests to temp directory
def test_mfa_aligner():
    # download dev-clean dataset from https://www.openslr.org/12 for tests to temp directory, if not already downloaded
    dataset_dir = Path(TEMP_DIR) / "LibriSpeech/dev-clean"
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

    # create dataset
    dataset = DirectoryDataset(dataset_dir)

    # create MFA aligner
    aligner = MFAAligner()
