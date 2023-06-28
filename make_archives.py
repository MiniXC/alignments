from pathlib import Path
import os
import tarfile
import shutil
import argparse
from tqdm.auto import tqdm

def tar_textgrids(path, archive_path):
    path = Path(path)
    archive_path = Path(archive_path)
    if archive_path.is_file():
        print(f"{archive_path} already exists. Skipping.")
        return
    with tarfile.open(archive_path, 'w:gz') as tar:
        for file in tqdm(path.glob('**/*.TextGrid')):
            tar.add(file, arcname=file.name)

if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--archive_path', type=str, required=True)
    args = parser.parse_args()
    # tar textgrids
    tar_textgrids(args.path, args.archive_path)