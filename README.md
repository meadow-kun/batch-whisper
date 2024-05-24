
# Whisper Batch Transcription Script

This script batch transcribes audio files in a directory to text with timestamps using the Whisper model.

## Features

- Transcribes `.mp3`, `.wav`, and `.m4a` audio files
- Saves transcriptions with timestamps to text files
- Skips already transcribed files to avoid redundant processing

## Requirements

- Python 3.7 or higher
- Whisper library

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/meadow-kun/batch-whisper.git
    cd whisper-transcription
    ```

2. **Create and activate a new conda environment:**

    ```bash
    conda create -n whisper-env python=3.12
    conda activate whisper-env
    ```

3. **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the script:**

    ```bash
    python transcribe.py /path/to/your/audio/files
    ```

## Usage

The script processes all audio files in the specified directory and generates transcripts with timestamps. Each transcript is saved in a separate directory named after the corresponding audio file, with the suffix `_transcripts`.

### Example

To transcribe all audio files in the `audio_files` directory:

```bash
python transcribe.py audio_files
```

## Notes

- Ensure you have sufficient disk space for the transcription outputs.
- The script will skip audio files that have already been transcribed.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements or bug fixes.
