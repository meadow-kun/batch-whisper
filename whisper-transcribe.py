import os
import glob
import argparse
import whisper

def transcribe_audio_to_text_with_timestamps(audio_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = whisper.load_model("medium.en")

    # Perform the transcription
    result = model.transcribe(audio_path)

    # Write the transcription with timestamps to a text file
    output_path = os.path.join(output_dir, "transcript.txt")
    with open(output_path, 'w', encoding='utf-8') as file:
        for segment in result['segments']:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            file.write(f"[{start:.2f} - {end:.2f}] {text}\n")

    print(f"Transcription with timestamps saved to {output_path}")

def process_directory(directory_path):
    # Find all audio files in the directory
    audio_files = glob.glob(os.path.join(directory_path, '*.mp3')) + \
                  glob.glob(os.path.join(directory_path, '*.wav')) + \
                  glob.glob(os.path.join(directory_path, '*.m4a'))

    for audio_path in audio_files:
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_dir = os.path.join(directory_path, base_filename + "_transcripts")
        output_file = os.path.join(output_dir, "transcript.txt")

        # Check if the output file already exists
        if not os.path.exists(output_file):
            transcribe_audio_to_text_with_timestamps(audio_path, output_dir)
        else:
            print(f"Transcription already exists for {audio_path}, skipping...")

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files to text with timestamps using Whisper.")
    parser.add_argument("directory_path", type=str, help="Directory containing audio files")
    args = parser.parse_args()

    process_directory(args.directory_path)

if __name__ == "__main__":
    main()
