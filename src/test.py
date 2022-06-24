from datasets.libritts import LibrittsDataset

if __name__ == "__main__":
    tts_ds = LibrittsDataset(
        "../data/train-clean-aligned",
        "../data/train-clean",
        "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
        verbose=True,
    )
    print(tts_ds.tokens)
    print(tts_ds.token_counts)
    # for item in tts_ds:
    #     print(item)