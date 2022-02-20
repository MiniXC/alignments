from pathlib import Path
import click
import subprocess
import re
import os
import shutil
from glob import glob
from rich import print
from rich.console import Console
from rich.progress import track

console = Console()

@click.group()
def cli():
    pass

@cli.command('create')
@click.argument(
    "source_directory",
    type=click.Path(exists=True, file_okay=False),
)
@click.argument(
    "target_directory",
    type=click.Path(exists=False, file_okay=False),
)
@click.option("--copy-wav", is_flag=True)
@click.option("--to-ipa", is_flag=True)
@click.option("--transcript-pattern", default='{parent}/{name}.normalized.txt')
@click.option("--speaker-pattern", default='.*[^\d](\d+)_\d+_\d+_\d+.wav')
def create(source_directory, target_directory, copy_wav, to_ipa, transcript_pattern, speaker_pattern):
    """
    Create TextGrid files for .wav and .lab files in SOURCE DIRECTORY in TARGET DIRECTORY.
    """
    results = []
    speaker_re = re.compile(speaker_pattern)
    for wav_file in track(list(Path(source_directory).rglob('*.wav')), description=f'finding wav files in {source_directory}'):
        speaker = speaker_re.match(str(wav_file)).group(1)
        transcript = transcript_pattern.format(parent=wav_file.parent, name=wav_file.name.replace(".wav", ""))
        results.append((wav_file, Path(transcript), speaker))
    speakers = set()
    if copy_wav:
        wav_action = "copying"
    else:
        wav_action = "symlinking"
    for wav, transcript, speaker in track(results, description=f'copying transcripts & {wav_action} wav files'):
        if speaker not in speakers:
            speakers.add(speaker)
            Path(target_directory, speaker).mkdir(parents=True, exist_ok=False)
        wav_target = Path(target_directory, speaker, wav.name)
        if not copy_wav:
            os.symlink(Path(wav).resolve(), wav_target)
        else:
            shutil.copy(wav, wav_target)
        shutil.copy(transcript, str(wav_target).replace(".wav", ".lab"))


def run_subprocess(command, desc, capture=True):
    with console.status(desc):
        out = subprocess.run(command, capture_output=capture)
    if out.returncode == 1:
        print(f'[red]✕[/red] {desc}')
        raise Exception(out.stderr.decode())
    else:
        print(f'[green]✓[/green] {desc}')
    return out

@cli.command('install')
@click.option("--verbose", is_flag=True)
def install(verbose):
    """
    Create new conda enviroment used for aligning.
    """
    out = run_subprocess(['conda', 'env', 'list', '--json'], "checking conda environments")
    if "envs/mfa\"" in out.stdout.decode():
        print("[yellow]\"mfa\" conda environment already exists, skipping installation.[/yellow]")
    else:
        out = run_subprocess(['conda', 'create', '-n', 'mfa', '-y'], 'creating "mfa" conda environment', not verbose)
        out = run_subprocess(['conda', 'install', 'montreal-forced-aligner', '-c', 'conda-forge', '-n', 'mfa', '-y'], 'installing montreal-forced-aligner', not verbose)


if __name__ == '__main__':
    cli()