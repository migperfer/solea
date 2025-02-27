import os
import json
import sys
import argparse

import pandas as pd
import soundfile as sf
from scipy.signal import resample
import yt_dlp
import numpy as np

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


def process_dali_group(group, root_folder, sample_rate, overwrite):
    """Process all chunks for a single dali_id."""
    dali_id = group.dali_id.iloc[0]
    youtube_id = group["youtube_id"].iloc[0]

    # Create directory for the song
    dali_folder = os.path.join(root_folder, dali_id)
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
        if original_rate != sample_rate:
            print(f"Resampling audio for {dali_id} from {original_rate} Hz to {sample_rate} Hz...")
            data = resample_audio(data, original_rate, sample_rate)

        # Process each chunk for this dali_id
        for _, row in group.iterrows():
            chunk_id = str(row["chunk_id"])
            beginning = float(row["beginning"])
            end = float(row["end"])

            # Create chunk folder
            chunk_folder = os.path.join(dali_folder, chunk_id)
            if not overwrite and os.path.exists(chunk_folder):
                print(f"Skipping existing chunk {chunk_id} for {dali_id}.")
                continue
            os.makedirs(chunk_folder, exist_ok=True)

            # Extract chunk and save as .flac
            chunk_audio = data[int(beginning * sample_rate):int(end * sample_rate)]
            chunk_audio_path = os.path.join(chunk_folder, "audio.flac")
            sf.write(chunk_audio_path, chunk_audio, sample_rate, format="FLAC")
            notes = np.array(row.notes)
            np.savetxt(os.path.join(chunk_folder, "notes.tsv"), notes, delimiter="\t", header="onset,offset,pitch")

        print(f"Processed all chunks for {dali_id}.")

    except Exception as e:
        print(f"Error processing {dali_id}: {e}")

    # Remove the downloaded full audio to save space
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Deleted full audio for {dali_id}.")


def process_json(json_path, root_folder, sample_rate, overwrite):
    """Process the JSON file."""
    # Load JSON into a pandas DataFrame
    with open(json_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Group by dali_id and process each group
    grouped = df.groupby("dali_id")
    for _, group in grouped:
        process_dali_group(group, root_folder, sample_rate, overwrite)


def check_dataset(json_path, root_folder):
    """Check the dataset for any missing files."""
    # Load JSON into a pandas DataFrame
    with open(json_path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)

    # Check each group
    grouped = df.groupby("dali_id")
    for dali_id, group in grouped:
        dali_folder = os.path.join(root_folder, dali_id)
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
    parser = argparse.ArgumentParser(description="Process and check DALI dataset.")
    parser.add_argument("--root_folder", type=str, default="./downloaded_songs", help="Root folder for downloaded songs")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--json_file_path", type=str, default="solea.json", help="Path to the JSON file")
    parser.add_argument("--check_dataset", action="store_true", help="Check the dataset for missing files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing chunk folders")

    args = parser.parse_args()

    if args.check_dataset:
        check_dataset(args.json_file_path, args.root_folder)
    else:
        process_json(args.json_file_path, args.root_folder, args.sample_rate, args.overwrite)