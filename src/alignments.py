import click
import subprocess
from rich import print
from rich.console import Console 

console = Console()

@click.group()
def cli():
    pass

@cli.command('create')
@click.argument(
    "source directory",
    type=click.Path(exists=True, file_okay=False),
)
@click.argument(
    "target directory",
    type=click.Path(exists=False, file_okay=False),
)
@click.option("--copy-wav", is_flag=True)
@click.option("--to-ipa", is_flag=True)
def create(source_dir, target_dir, copy_wav, to_ipa):
    """
    Create TextGrid files for .wav and .lab files in SOURCE DIRECTORY in TARGET DIRECTORY.
    """
    pass

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
    out = run_subprocess(['conda', 'env', 'list', '--json'], "checking conda environments", verbose)
    if "envs/mfa\"" in out.stdout.decode():
        print("[yellow]\"mfa\" conda environment already exists, skipping creation.[/yellow]")
    else:
        out = run_subprocess(['conda', 'create', '-n', 'mfa', '-y'], 'creating "mfa" conda environment', verbose)
        out = run_subprocess(['conda', 'install', 'montreal-forced-aligner', '-c', 'conda-forge', '-n', 'mfa', '-y'], 'installing montreal-forced-aligner', verbose)


if __name__ == '__main__':
    cli()