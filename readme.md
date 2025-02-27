# Instructions

1. Install the requirements `pip install requirements.txt`
2. Download the dataset with `python solea_downloader.py`

```bash
usage: solea_downloader.py [-h] [--root_folder ROOT_FOLDER] [--sample_rate SAMPLE_RATE] [--json_file_path JSON_FILE_PATH] [--check_dataset] [--overwrite]

Process and check SOLEA dataset.

options:
  -h, --help            show this help message and exit
  --root_folder ROOT_FOLDER
                        Root folder for downloaded songs
  --sample_rate SAMPLE_RATE
                        Target sample rate
  --json_file_path JSON_FILE_PATH
                        Path to the JSON file
  --check_dataset       Check the dataset for missing files
  --overwrite           Overwrite existing chunk folders

```


The dataset will be downloaded with the following structure:
```txt
ROOT_FOLDER/
  dali_id/
    chunk_id/
      audio.flac
      notes.tsv
```