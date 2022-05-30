from pathlib import Path
from tabnanny import verbose
import click
import subprocess
import re
import os
import shutil
import multiprocessing

from rich import print
from rich.console import Console
from rich.progress import track

console = Console()


@click.group()
def cli():
    pass


@cli.command("create")
@click.argument(
    "source_directory", type=click.Path(exists=True, file_okay=False),
)
@click.argument(
    "target_directory", type=click.Path(exists=False, file_okay=False),
)
@click.option("--copy-wav", is_flag=True)
@click.option("--overwrite", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--transcript-pattern", default="{parent}/{name}.normalized.txt")
@click.option("--speaker-pattern", default=".*[^\d](\d+)_\d+_\d+_\d+.wav")
@click.option("--acoustic-model", default="english_us_arpa")
@click.option(
    "--lexicon",
    default="https://www.openslr.org/resources/11/librispeech-lexicon.txt",
    help="url or path to the lexicon",
)
def create(
    source_directory,
    target_directory,
    copy_wav,
    overwrite,
    transcript_pattern,
    speaker_pattern,
    lexicon,
    acoustic_model,
    verbose,
):
    """
    Create TextGrid files for .wav and .lab files in SOURCE DIRECTORY in TARGET DIRECTORY.
    """
    if (
        not overwrite
        and Path(target_directory).exists()
        and any(os.scandir(target_directory))
    ):
        raise ValueError(
            "TARGET_DIRECTORY is not empty, use --overwrite or specify empty or new directory"
        )
    if (
        overwrite
        and Path(target_directory).exists()
        and any(os.scandir(target_directory))
    ):
        shutil.rmtree(target_directory)
    results = []
    speaker_re = re.compile(speaker_pattern)
    for wav_file in track(
        list(Path(source_directory).rglob("*.wav")),
        description=f"finding wav files in {source_directory}",
    ):
        speaker = speaker_re.match(str(wav_file)).group(1)
        transcript = transcript_pattern.format(
            parent=wav_file.parent, name=wav_file.name.replace(".wav", "")
        )
        results.append((wav_file, Path(transcript), speaker))
    speakers = set()
    if copy_wav:
        wav_action = "copying"
    else:
        wav_action = "symlinking"
    for wav, transcript, speaker in track(
        results, description=f"copying transcripts & {wav_action} wav files"
    ):
        if speaker not in speakers:
            speakers.add(speaker)
            Path(target_directory, speaker).mkdir(parents=True, exist_ok=False)
        wav_target = Path(target_directory, speaker, wav.name)
        if not copy_wav:
            os.symlink(Path(wav).resolve(), wav_target)
        else:
            shutil.copy(wav, wav_target)
        shutil.copy(transcript, str(wav_target).replace(".wav", ".lab"))
    if "http://" in lexicon or "https://" in lexicon:
        run_subprocess(f"wget {lexicon}", "downloading lexicon", cwd=target_directory)
        lexicon_path = os.path.join(target_directory, lexicon.split("/")[-1])
    else:
        lexicon_path = lexicon
    target_temp_directory = "/tmp/aligments_temp/textgrids"
    if Path(target_temp_directory).exists():
        shutil.rmtree(target_temp_directory)
    Path(target_temp_directory).parent.mkdir(parents=True, exist_ok=True)
    run_subprocess(
        f". $CONDA_PREFIX/etc/profile.d/conda.sh \
            && conda activate mfa \
            && mfa validate {target_directory} {lexicon_path} {acoustic_model} -j {multiprocessing.cpu_count()} \
            && mfa align {target_directory} {lexicon_path} {acoustic_model} {target_temp_directory} -j {multiprocessing.cpu_count()} --clean",
        "running montreal forced aligner",
        not verbose,
    )
    run_subprocess(
        f"cp -rT {target_temp_directory} {target_directory}",
        "copying TextGrids to TARGET_DIRECTORY",
    )


def run_subprocess(command, desc, capture=True, cwd=None):
    with console.status(desc):
        out = subprocess.run(command, capture_output=capture, cwd=cwd, shell=True)
    if out.returncode != 0:
        print(f"[red]✕[/red] {desc}")
        raise Exception(out.stderr.decode())
    else:
        print(f"[green]✓[/green] {desc}")
    return out


@cli.command("install")
@click.option("--verbose", is_flag=True)
def install(verbose):
    """
    Create new conda enviroment used for aligning.
    """
    out = run_subprocess("conda env list --json", "checking conda environments")
    if 'envs/mfa"' in out.stdout.decode():
        print(
            '[yellow]"mfa" conda environment already exists, skipping installation.[/yellow]'
        )
    else:
        out = run_subprocess(
            "conda create -n mfa -y", 'creating "mfa" conda environment', not verbose
        )
        out = run_subprocess(
            "conda install montreal-forced-aligner -c conda-forge -n mfa -y",
            "installing montreal-forced-aligner",
            not verbose,
        )


if __name__ == "__main__":
    cli()
