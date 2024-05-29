
# Whisper Batch Transcription Tool

## Overview

This Python program batch transcribes audio files to text with timestamps using OpenAI's Whisper model. It supports parallel processing to utilize multiple CPU cores, improving the efficiency of transcribing large batches of audio files.

## Features

- Batch processing of audio files in a specified directory
- Supports various audio file formats (MP3, WAV, M4A)
- Outputs transcriptions with timestamps
- Utilizes multiprocessing for faster processing
- Verbose logging for detailed progress tracking

## Requirements

- Python 3.7 or higher
- Required Python packages: `openai-whisper`, `tqdm`, `multiprocessing`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/meadow-kun/whisper-batch-transcription.git
    cd whisper-batch-transcription
    ```

2. Install the required Python packages:
    ```bash
    pip install openai-whisper tqdm
    ```

## Usage

1. Prepare a directory containing the audio files you want to transcribe.
2. Run the script with the directory path as an argument:
    ```bash
    python transcribe.py /path/to/your/audio/files
    ```
3. Use the `-v` or `--verbose` flag for detailed logging:
    ```bash
    python transcribe.py /path/to/your/audio/files -v
    ```

## Script Details

### transcribe.py

```python
import os
import glob
import argparse
import whisper
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model globally
model = None

def load_model(verbose=False):
    global model
    if model is None:
        if verbose:
            logging.info("Loading Whisper model...")
        model = whisper.load_model("large")

def transcribe_audio_to_text_with_timestamps(audio_path, output_dir, verbose=False):
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load the model
        load_model(verbose)

        # Perform the transcription
        if verbose:
            logging.info(f"Transcribing {audio_path}...")
        result = model.transcribe(audio_path, language="sv", verbose=True)

        # Write the transcription with timestamps to a text file
        output_path = os.path.join(output_dir, "transcript.txt")
        if verbose:
            logging.info(f"Saving transcription to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as file:
            for segment in result['segments']:
                start = segment['start']
                end = segment['end']
                text = segment['text']
                file.write(f"[{start:.2f} - {end:.2f}] {text}\n")

        if verbose:
            logging.info(f"Transcription with timestamps saved to {output_path}")
    except Exception as e:
        logging.error(f"Error transcribing {audio_path}: {e}")

def process_file(args):
    audio_path, directory_path, verbose = args
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.join(directory_path, base_filename + "_transcripts")
    output_file = os.path.join(output_dir, "transcript.txt")

    # Check if the output file already exists
    if not os.path.exists(output_file):
        transcribe_audio_to_text_with_timestamps(audio_path, output_dir, verbose)
    else:
        if verbose:
            logging.info(f"Transcription already exists for {audio_path}, skipping...")

def process_directory(directory_path, verbose=False):
    # Find all audio files in the directory
    audio_files = glob.glob(os.path.join(directory_path, '*.mp3')) +                   glob.glob(os.path.join(directory_path, '*.wav')) +                   glob.glob(os.path.join(directory_path, '*.m4a'))

    # Prepare arguments for parallel processing
    args = [(audio_path, directory_path, verbose) for audio_path in audio_files]

    # Limit the number of workers to avoid overloading the system
    num_workers = min(max(1, cpu_count() // 4), len(audio_files))
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_file, args), total=len(audio_files), desc="Processing audio files", unit="file"))

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files to text with timestamps using Whisper.")
    parser.add_argument("directory_path", type=str, help="Directory containing audio files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    process_directory(args.directory_path, args.verbose)

if __name__ == "__main__":
    main()
```

## Notes

- Ensure your audio files are of high quality for better transcription accuracy.
- The Whisper model can be memory intensive. Monitor your system's resources during execution and adjust the number of worker processes if necessary.

## License

This project is licensed under the MIT License.

## Author

Created by [meadow-kun](https://github.com/meadow-kun). Feel free to contribute or raise issues on the repository.
