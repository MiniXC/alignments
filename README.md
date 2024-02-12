# Alignments 1.0
![Coverage](badges/coverage.svg)
![Tests](badges/tests.svg)

This package is a wrapper around the Montreal Forced Aligner. It provides a simple interface for aligning audio files with their corresponding transcripts.

## Why not use the Montreal Forced Aligner directly?
The Montreal Forced Aligner is a powerful tool, but it can be difficult to use. This package provides a simple interface for aligning audio files with their corresponding transcripts in python.

## Example Usage
```python
from alignments import Aligner

aligner = Aligner('path/to/montreal-forced-aligner')
aligner._align('path/to/audio.wav', 'path/to/transcript.txt', 'alignment_output_dir')
```

## Installation
This package uses montreal-forced-aligner 3.0.0 which depends on a conda install of kalpy. To install the package, run the following command:
```bash
conda install kalpy=0.5.9
```

Then, install the package using pip:
```bash
pip install alignments==1.0.0a1
```