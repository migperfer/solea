import os
import json
import sys

import pandas as pd
import soundfile as sf
from scipy.signal import resample
import yt_dlp
import numpy as np

# Define paths and constants
ROOT_FOLDER = "./downloaded_songs"
SAMPLE_RATE = 16000  # Target sample rate

# Create ROOT_FOLDER if it doesn't exist
os.makedirs(ROOT_FOLDER, exist_ok=True)


def download_youtube_audio(youtube_id, output_path):
    """Download only audio from YouTube using yt-dlp."""
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': os.path.join(output_path, f"{youtube_id}"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'flac',
        }],
        'quiet': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
        return os.path.join(output_path, f"{youtube_id}.flac")
    except Exception as e:
        print(f"Error downloading video {youtube_id}: {e}")
        return None


def resample_audio(audio, original_rate, target_rate):
    """Resample audio to the target sample rate."""
    num_samples = int(len(audio) * target_rate / original_rate)
    return resample(audio, num_samples)


def process_dali_group(group):
    """Process all chunks for a single dali_id."""
    dali_id = group.dali_id.iloc[0]
    youtube_id = group["youtube_id"].iloc[0]

    # Create directory for the song
    dali_folder = os.path.join(ROOT_FOLDER, dali_id)
    os.makedirs(dali_folder, exist_ok=True)

    audio_path = os.path.join(dali_folder, f"{youtube_id}.webm")

    # Download the audio for this dali_id if not already downloaded
    if not os.path.exists(audio_path):
        print(f"Downloading audio for {dali_id} (YouTube ID: {youtube_id})...")
        audio_path = download_youtube_audio(youtube_id, dali_folder)
        if not audio_path:
            print(f"Skipping {dali_id} due to download failure.")
            return

    try:
        # Load the audio
        data, original_rate = sf.read(audio_path)

        # Resample to the target rate if necessary
        if original_rate != SAMPLE_RATE:
            print(f"Resampling audio for {dali_id} from {original_rate} Hz to {SAMPLE_RATE} Hz...")
            data = resample_audio(data, original_rate, SAMPLE_RATE)

        # Process each chunk for this dali_id
        for _, row in group.iterrows():
            chunk_id = str(row["chunk_id"])
            beginning = float(row["beginning"])
            end = float(row["end"])

            # Create chunk folder
            chunk_folder = os.path.join(dali_folder, chunk_id)
            os.makedirs(chunk_folder, exist_ok=True)

            # Extract chunk and save as .flac
            chunk_audio = data[int(beginning * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            chunk_audio_path = os.path.join(chunk_folder, "audio.flac")
            sf.write(chunk_audio_path, chunk_audio, SAMPLE_RATE, format="FLAC")
            notes = np.array(row.notes)
            np.savetxt(os.path.join(chunk_folder, "notes.tsv"), notes, delimiter="\t", header="onset,offset,pitch")

        print(f"Processed all chunks for {dali_id}.")

    except Exception as e:
        print(f"Error processing {dali_id}: {e}")

    # Remove the downloaded full audio to save space
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Deleted full audio for {dali_id}.")


def process_json(json_path):
    """Process the JSON file."""
    # Load JSON into a pandas DataFrame
    with open(json_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Group by dali_id and process each group
    grouped = df.groupby("dali_id")
    for _, group in grouped:
        process_dali_group(group)


def check_dataset(json_path):
    """Check the dataset for any missing files."""
    # Load JSON into a pandas DataFrame
    with open(json_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Check each group
    grouped = df.groupby("dali_id")
    for dali_id, group in grouped:
        dali_folder = os.path.join(ROOT_FOLDER, dali_id)
        for _, row in group.iterrows():
            chunk_id = str(row["chunk_id"])
            chunk_folder = os.path.join(dali_folder, chunk_id)
            chunk_audio_path = os.path.join(chunk_folder, "audio.flac")
            chunk_notes_path = os.path.join(chunk_folder, "notes.tsv")
            if not os.path.exists(chunk_audio_path):
                # Write in a file which chunks are missing
                with open("missing_chunks.txt", "a") as f:
                    f.write(f"Missing audio for {dali_id} chunk {chunk_id}\n")
            if not os.path.exists(chunk_notes_path):
                with open("missing_notes.txt", "a") as f:
                    f.write(f"Missing notes for {dali_id} chunk {chunk_id}\n")


if __name__ == "__main__":
    json_file_path = "solea.json"  # Replace with the actual path to your JSON file
    # If the argument is --check_dataset, only check the dataset
    if "--check_dataset" in sys.argv:
        check_dataset(json_file_path)
        sys.exit()
    process_json(json_file_path)
