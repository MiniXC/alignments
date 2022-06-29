from abc import abstractmethod
import abc
import imp
from pathlib import Path
from string import punctuation
from urllib import request
from zipfile import ZipFile
import tarfile
import os, shutil
import subprocess
import platform
import multiprocessing
import unicodedata
import warnings

from torch.utils.data import Dataset
from tqdm.rich import tqdm
from tqdm.contrib.concurrent import process_map
from rich import print
from rich.console import Console
import textgrid

console = Console()
warnings.filterwarnings("ignore", message="rich is experimental/alpha")

class DownloadProgressBar():
    def __init__(self):
        self.pbar = None
        self.downloaded = 0
        self.sum = 0

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=tqdm(total=total_size, desc="Downloading")

        downloaded = block_num * block_size

        self.pbar.update(downloaded-self.downloaded)
        self.downloaded = downloaded
        if self.downloaded >= total_size:
            self.pbar.close()
            self.pbar = None

def run_subprocess(command, desc, capture=True, cwd=None):
    if capture:
        with console.status(desc):
            out = subprocess.run(command, capture_output=capture, cwd=cwd, shell=True)
    else:
        print(f"[blue]⏱[/blue] {desc}")
        out = subprocess.run(command, capture_output=capture, cwd=cwd, shell=True)
    if out.returncode != 0:
        print(f"[red]✕[/red] {desc}")
        if out.stderr is not None:
            raise Exception(out.stderr.decode())
    else:
        print(f"[green]✓[/green] {desc}")
    return out

def check_install_mfa(verbose, force):
    """
    Create new conda enviroment used for aligning.
    """
    out = run_subprocess("conda env list --json", "checking conda environments")
    env_out = out.stdout.decode()
    if force and 'envs/alignments_mfa"' in env_out:
        out = run_subprocess(
            f"conda env remove -n alignments_mfa", 'removing "alignments_mfa" conda environment', not verbose
        )
    if 'envs/alignments_mfa"' not in env_out or force:
        out = run_subprocess(
            f"conda create -y -n alignments_mfa python={platform.python_version()}", 'creating "alignments_mfa" conda environment', not verbose
        )
        out = run_subprocess(
            "conda install montreal-forced-aligner -c conda-forge -n alignments_mfa -y",
            "installing montreal-forced-aligner",
            not verbose,
        )

class AlignmentDataset(Dataset):
    def __init__(
        self,
        target_directory,
        source_directory=None,
        source_url=None,
        force="none", # "none", "all", "download", "processing", "lexicon", "validation", "alignment"
        symbolic_links=True,
        acoustic_model="english_us_arpa",
        g2p_model="english_us_arpa",
        lexicon=None,
        verbose=False,
        show_warnings=False,
        punctuation_marks="!?.,;",
    ):
        super().__init__()
        __metaclass__ = abc.ABCMeta
        self.target_directory = target_directory
        self.source_directory = source_directory
        self.source_url = source_url
        self.show_warnings = show_warnings
        self.punctuation_marks = punctuation_marks

        if source_directory is None:
            # skip all other init steps
            self._load_files()
            return

        if len(list(Path(target_directory).glob("**/*.TextGrid"))) > 0 and force == "none":
            print(f"[green]✓[/green] {target_directory} already contains TextGrids")
            self._load_files()
            return

        # DOWNLOAD
        if self.source_url is not None:
            if force == "download" or force == "all":
                shutil.rmtree(source_directory)
            if not Path(source_directory).exists():
                download_path = Path("/tmp/alignments/downloads")
                download_path.mkdir(exist_ok=True, parents=True)
                if self.source_url.endswith(".zip"):
                    tmp_path = download_path / "data.zip"
                    response = request.urlretrieve(self.source_url, tmp_path, DownloadProgressBar())
                    ZipFile(tmp_path).extractall(source_directory)
                elif self.source_url.endswith(".tar.gz"):
                    tmp_path = download_path / "data.tar.gz"
                    response = request.urlretrieve(self.source_url, tmp_path, DownloadProgressBar())
                    tarfile.open(tmp_path).extractall(source_directory)
                else:
                    raise ValueError("Unknown file type, only .zip and .tar.gz are supported.")
            else:
                print("Source directory already exists. Skipping [blue]download[/blue].")

        # LOAD
        if force == "processing" or force == "all":
            shutil.rmtree(target_directory)
        if not Path(target_directory).exists():
            Path(target_directory).mkdir(exist_ok=True, parents=True)
            for item in self.collect_data(self.source_directory):
                if not item["path"].suffix == ".wav":
                    raise ValueError("Only .wav files are supported.")
                target_path = (Path(target_directory) / item["speaker"] / item["path"].name)
                target_path.parent.mkdir(exist_ok=True, parents=True)
                if symbolic_links:
                    target_path.resolve().symlink_to(item["path"].resolve())
                else:
                    shutil.copy(item["path"], target_path)
                target_path.with_suffix(".lab").write_text(item["transcript"])
        else:
            print("Target directory already exists. Skipping [blue]processing[/blue].")

        # PREPARE
        check_install_mfa(True, force=="conda" or force=="all")
        download_command = f". $CONDA_PREFIX/etc/profile.d/conda.sh \
                            && conda activate alignments_mfa \
                            && mfa model download acoustic {acoustic_model}"
        if g2p_model is not None:
            download_command += f" && mfa model download g2p {g2p_model}"
        run_subprocess(
            download_command,
            "downloading necessary models",
            not verbose
        )

        # LEXICON
        lexicon_path = Path(source_directory) / "lexicon.txt"
        if force == "lexicon" or force == "all":
            lexicon_path.unlink(missing_ok=True)
        if not lexicon_path.exists():
            if lexicon is not None:
                if lexicon.startswith("http"):
                    response = request.urlretrieve(lexicon, lexicon_path, DownloadProgressBar())
            elif g2p_model is not None:
                g2p_command = f". $CONDA_PREFIX/etc/profile.d/conda.sh \
                                && conda activate alignments_mfa \
                                && mfa g2p {g2p_model} {target_directory} {lexicon_path} -j {multiprocessing.cpu_count()}"
                run_subprocess(
                    g2p_command,
                    "creating lexicon using g2p model (this could take a while)",
                    not verbose
                )
        else:
            print("Lexicon already exists. Skipping [blue]lexicon[/blue] creation.")

        # VALIDATE
        lexicon_with_oov_path = Path(source_directory) / "lexicon_with_oov.txt"
        oov_path = Path(os.environ["MFA_ROOT_DIR"]) / f"{Path(target_directory).name}_validate_pretrained" / "oovs_found_lexicon.txt"
        if force == "validation" or force == "all":
            lexicon_with_oov_path.unlink(missing_ok=True)
            shutil.rmtree(oov_path.parent)
        if not lexicon_with_oov_path.exists() or not oov_path.parent.exists():
            align_command = f". $CONDA_PREFIX/etc/profile.d/conda.sh \
                                && conda activate alignments_mfa \
                                && mfa validate {target_directory} {lexicon_path} {acoustic_model} -j {multiprocessing.cpu_count()} --clean --overwrite"
            run_subprocess(
                align_command,
                "validating data",
                not verbose
            )
            lexicon_tmp_path = Path("/tmp/alignments") / "lexicon.txt"
            g2p_command = f". $CONDA_PREFIX/etc/profile.d/conda.sh \
                                && conda activate alignments_mfa \
                                && mfa g2p {g2p_model} {oov_path} {lexicon_tmp_path} -j {multiprocessing.cpu_count()}"
            run_subprocess(
                g2p_command,
                "using g2p model for oovs",
                not verbose
            )
            lexicon_with_oov_path.write_text(lexicon_path.read_text()+lexicon_tmp_path.read_text())
        else:
            print("Lexicon with OOV words and valid. directory already exists. Skipping [blue]validation[/blue].")
        

        # ALIGN
        if force == "alignment" or force == "all":
            for textgrid in Path(target_directory).glob("**/*.TextGrid"):
                textgrid.unlink(missing_ok=True)
        if len(list(Path(target_directory).glob("**/*.TextGrid"))) == 0:
            target_temp_directory = Path("/tmp/alignments/alignments")
            shutil.rmtree(target_temp_directory, ignore_errors=True)
            target_temp_directory.mkdir(exist_ok=True, parents=True)
            align_command = f". $CONDA_PREFIX/etc/profile.d/conda.sh \
                                    && conda activate alignments_mfa \
                                    && mfa align {target_directory} {lexicon_with_oov_path} {acoustic_model} {target_temp_directory} -j {multiprocessing.cpu_count()} --clean --overwrite"
            run_subprocess(
                    align_command,
                    "aligning data",
                    not verbose
                )
            run_subprocess(
                f"cp -rT {target_temp_directory} {target_directory}",
                "copying TextGrids to target directory",
            )
        else:
            print("TextGrids already exist. Skipping [blue]alignment[/blue].")

        self._load_files()
        
    def _load_files(self):
        """
        Loads the files from the source directory.
        """
        self.files = []
        self.missing = 0
        for item in Path(self.target_directory).glob("**/*.wav"):
            if item.with_suffix(".TextGrid").exists() and item.with_suffix(".lab").exists():
                self.files.append([item, item.with_suffix(".TextGrid"), item.with_suffix(".lab")])
            else:
                self.missing += 1
        print(f"Found {len(self.files)} files with {self.missing} missing.")
        self.data = []
        self.tokens = set()
        self.token_counts = {}
        none_count = 0
        for item in process_map(
                self._create_item,
                self.files,
                chunksize=100,
                max_workers=multiprocessing.cpu_count(),
                desc="collecting textgrid and audio files",
                tqdm_class=tqdm,
            ):
            if "incorrect" not in item:
                self.data.append(item)
                tokens = [x[2] for x in item["phones"]]
                self.tokens.update(tokens)
                for token in tokens:
                    if token not in self.token_counts:
                        self.token_counts[token] = 1
                    else:
                        self.token_counts[token] += 1
            else:
                if self.show_warnings:
                    print(f"WARNING: \"{item['text']}\" is incorrect and was skipped because {item['incorrect']}")
                none_count += 1
        print(f"Found {len(self.data)} items with {none_count} skipped due to bad punctuation.")

    @abstractmethod
    def collect_data(self, directory):
        """
        Expects a list of dictionaries with the following keys:
        - "path": the path to the wav file
        - "speaker": the speaker id
        - "transcript": the transcript
        """
        raise NotImplementedError()

    def _create_item(self, file):
        # TODO: fix quotes and triple dots
        wav, grid, lab = file
        text = Path(lab).read_text().lower()
        words = [
            x.replace('"', '')[1:] if x.replace('"', '').startswith("'") else x.replace('"', '') 
            for x in Path(lab).read_text().lower().split()
        ]
        last_word = 0
        for i, word in enumerate(words):
            has_alnum = any([x.isalnum() for x in word])
            if not has_alnum:
                if i > 0:
                    words[last_word] = words[last_word] + words[i]
                words[i] = ''
            else:
                last_word = i
        words = [x for x in words if len(x) > 0]
        file = textgrid.TextGrid.fromFile(grid)
        marks = [x for x in file[0]]
        punctuations = []
        mark_i = 0
        for word in words:
            if mark_i >= len(marks):
                return {
                    "incorrect": "marks out of range",
                    "text": text,
                }
            current_mark = marks[mark_i]
            while len(current_mark.mark) == 0:
                mark_i += 1
                if mark_i >= len(marks):
                    return {
                        "incorrect": "marks out of range",
                        "text": text,
                    }
                current_mark = marks[mark_i]
            if word.startswith(current_mark.mark):
                mark_i += 1
                punctuation = word[len(current_mark.mark):].replace("'", "").replace('"', '').replace('...', '')
                has_alnum = any([x.isalnum() for x in punctuation])
                if len(punctuation) >= 1 and not has_alnum and punctuation[0] in self.punctuation_marks:
                    punctuation = "[" + unicodedata.name(punctuation[0]) + "]"
                elif len(punctuation) == 0 or punctuation[0] not in self.punctuation_marks:
                    punctuation = "[SILENCE]"
                else:
                    return {
                        "incorrect": "word starts with punctuation",
                        "text": text,
                    }
                punctuations.append((current_mark.maxTime, punctuation))
            else:
                return {
                    "incorrect": "word does not start with mark",
                    "text": text,
                }
        phones = []
        phone_grid = file[1]
        max_time = phone_grid[-1].maxTime
        filter_grid = [x for x in phone_grid if len(x.mark) > 0]
        punc_i = 0
        for i, phone in enumerate(filter_grid):
            if i == 0:
                phones.append((0.0, phone.minTime, "[SILENCE]"))
            phones.append((phone.minTime, phone.maxTime, phone.mark))
            if punc_i < len(punctuations) and phone.maxTime == punctuations[punc_i][0]:
                if i < len(filter_grid) - 1:
                    next_time = filter_grid[i + 1].minTime
                else:
                    next_time = max_time
                phones.append((phone.maxTime, next_time, punctuations[punc_i][1]))
                punc_i += 1
        return {
            "wav": wav,
            "speaker": Path(wav).parent,
            "transcript": " ".join(words),
            "phones": phones
        }

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)