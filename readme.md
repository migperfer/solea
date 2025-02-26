# Instructions

1. Install the requirements `pip install requirements.txt`
2. Modify the `ROOT_FOLDER` and `SAMPLE_RATE` according to your preferences. The audios will be downloaded in that folder at the specified sample rate
3. Download the dataset with `python solea_downloader.py`

The dataset will be downloaded with the following structure:
```txt
ROOT_FOLDER/
  dali_id/
    chunk_id/
      audio.flac
      notes.tsv
```