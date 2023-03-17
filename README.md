# <img src="https://user-images.githubusercontent.com/3523501/225879662-4745f31a-6023-494d-a7a8-56d62d3a4aae.png" width="300" alt="alignments"/>
This tool is an abstraction of the [Montreal Forced Aligner](montreal-forced-aligner.readthedocs.io/) so it can be used as a PyTorch dataset.

```python
libritts_100 = LibrittsDataset(
  target_directory="../data/libritts-train-clean-100-aligned",
  source_directory="../data/LibriTTS/train-clean-100",
  source_url="https://www.openslr.org/resources/60/train-clean-100.tar.gz",
  chunk_size=10_000,
)
```

- ``source_directory`` specifies the directory where a) the data is already present or b) you want the data to be downloaded to
- ``target_directory`` specifies the directory you want the aligned data to be stored at

The dataset can then be used as follows:

```python
for item in libritts_100:
  item["wav"] # the audio
  item["speaker"] # speaker key
  item["transcript"] # normalized transcript
  item["phones"] # a list of triples (start_time_in_seconds, end_time_in_seconds, phone)
```

The ``"phones"`` list also inclodes ``[SILENCE]`` tokens between words, which are set to a length of 0 if no silence is present. In the case of punctuation, this silence token is replaced with the corresponding punctuation token.


## Supported Datasets

 - [x] LibriTTS
 - [ ] LJSpeech
 - [ ] CommonVoice
 - [ ] GlobalPhone

## Features

- Automatically downloads data on first run.
- Automatically downloads and installs Montreal Forced Aligner in its own conda environment.
- Symlinks audio files rather than copying them for alignment.
- Adds OOV words to Lexicon.
- Easily add your own dataset by extending ``AlignmentsDataset`` class and just implementing one method for collecting the transcripts.

## Planned Features

The following features are planned in future releases, please feel free to open issues if you have further ideas.

- [ ] Visualise Alignments in a similar style to [Praat](https://www.fon.hum.uva.nl/praat/)
- [ ] Integrate with [phones](github.com/MiniXC/phones) to allow automatic conversion to IPA phones


