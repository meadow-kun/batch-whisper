import os
import glob
import argparse
import whisper
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_audio_to_text_with_timestamps(audio_path, output_dir, verbose=False):
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load the model
        if verbose:
            logging.info(f"Loading Whisper model for {audio_path}...")
        model = whisper.load_model("medium.en")

        # Perform the transcription
        if verbose:
            logging.info(f"Transcribing {audio_path}...")
        result = model.transcribe(audio_path)

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
    audio_files = glob.glob(os.path.join(directory_path, '*.mp3')) + \
                  glob.glob(os.path.join(directory_path, '*.wav')) + \
                  glob.glob(os.path.join(directory_path, '*.m4a'))

    # Prepare arguments for parallel processing
    args = [(audio_path, directory_path, verbose) for audio_path in audio_files]

    # Limit the number of workers to avoid overloading the system
    num_workers = min(cpu_count() // 2, len(audio_files))
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
