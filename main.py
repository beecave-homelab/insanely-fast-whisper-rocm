import os
import time
import logging
from dotenv import load_dotenv
import argparse
import json
import subprocess
import sys

# Load .env file if it exists
load_dotenv()

# Default values
DEFAULT_UPLOADS = os.getenv("UPLOADS", "/app/uploads")
DEFAULT_TRANSCRIPTS = os.getenv("TRANSCRIPTS", "/app/transcripts")
DEFAULT_LOGS = os.getenv("LOGS", "/app/logs")
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
DEFAULT_VERBOSE = os.getenv("VERBOSE", "true").lower() in ("true", "1", "yes")
DEFAULT_MODEL = os.getenv("MODEL", "openai/whisper-large-v3")
DEFAULT_PROCESSED_TXT_DIR = os.getenv("PROCESSED_TXT_DIR", "transcripts-txt")
DEFAULT_PROCESSED_SRT_DIR = os.getenv("PROCESSED_SRT_DIR", "transcripts-srt")

class TxtFormatter:
    @classmethod
    def preamble(cls):
        return ""

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk['text']
        return f"{text}\n"

class SrtFormatter:
    @classmethod
    def preamble(cls):
        return ""

    @classmethod
    def format_seconds(cls, seconds):
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk['text']
        start, end = chunk['timestamp'][0], chunk['timestamp'][1]
        start_format, end_format = cls.format_seconds(start), cls.format_seconds(end)
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"

class VttFormatter:
    @classmethod
    def preamble(cls):
        return "WEBVTT\n\n"

    @classmethod
    def format_seconds(cls, seconds):
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk['text']
        start, end = chunk['timestamp'][0], chunk['timestamp'][1]
        start_format, end_format = cls.format_seconds(start), cls.format_seconds(end)
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"

def convert(input_path, output_format, output_dir, verbose):
    with open(input_path, 'r') as file:
        data = json.load(file)

    formatter_class = {
        'srt': SrtFormatter,
        'vtt': VttFormatter,
        'txt': TxtFormatter
    }.get(output_format)

    string = formatter_class.preamble()
    for index, chunk in enumerate(data['chunks'], 1):
        entry = formatter_class.format_chunk(chunk, index)

        if verbose:
            print(entry)

        string += entry

    with open(os.path.join(output_dir, f"output.{output_format}"), 'w', encoding='utf-8') as file:
        file.write(string)

def setup_logging(verbose):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f"{DEFAULT_LOGS}/script.log"),
                            logging.StreamHandler()
                        ])

def error_exit(message):
    logging.error(message)
    exit(1)

def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

def run_command_and_log(command, log_file_path):
    try:
        # Run the command and capture output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        with open(log_file_path, 'w') as log_file:
            for line in process.stdout:
                # Write to log file
                log_file.write(line)
                # Also print to console
                print(line, end='')
                sys.stdout.flush()  # Ensure output is printed immediately
        
        # Wait for the process to complete
        process.wait()
        
        return process.returncode
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description='Monitors a directory for file changes and processes files.')
    parser.add_argument('-u', '--uploads', default=DEFAULT_UPLOADS, help='Directory to watch for file changes')
    parser.add_argument('-t', '--transcripts', default=DEFAULT_TRANSCRIPTS, help='Directory to save transcript files')
    parser.add_argument('-l', '--logs', default=DEFAULT_LOGS, help='Directory to save log files')
    parser.add_argument('-b', '--batch-size', default=DEFAULT_BATCH_SIZE, type=int, help='Batch size for processing')
    parser.add_argument('-v', '--verbose', action='store_true', default=DEFAULT_VERBOSE, help='Enable verbose logging')
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, help='Model to be used for processing')
    parser.add_argument('--processed-txt-dir', default=DEFAULT_PROCESSED_TXT_DIR, help='Directory to save processed txt files')
    parser.add_argument('--processed-srt-dir', default=DEFAULT_PROCESSED_SRT_DIR, help='Directory to save processed srt files')
    parser.add_argument('--convert', action='store_true', help='Convert JSON to an output format')
    parser.add_argument('--input-file', help='Input JSON file path for conversion')
    parser.add_argument('--output-format', default='srt', help='Format of the output file (default: srt)', choices=['txt', 'vtt', 'srt'])
    parser.add_argument('--output-dir', default='.', help='Directory where the output file/s is/are saved')

    args = parser.parse_args()

    setup_logging(args.verbose)

    if args.convert:
        if not args.input_file:
            error_exit("Input file must be specified for conversion.")
        convert(args.input_file, args.output_format, args.output_dir, args.verbose)
        return

    # Create necessary directories
    create_directories([args.uploads, args.transcripts, args.logs, args.processed_txt_dir, args.processed_srt_dir])

    while True:
        files_found = False
        no_new_files_logged = False

        for filename in os.listdir(args.uploads):
            filepath = os.path.join(args.uploads, filename)

            if os.path.isfile(filepath):
                files_found = True
                transcript_output = os.path.join(args.transcripts, f"{os.path.splitext(filename)[0]}.json")
                log_file_path = os.path.join(args.logs, f"{os.path.splitext(filename)[0]}.log")

                # Check if the transcript already exists
                if os.path.exists(transcript_output):
                    time.sleep(60)
                    continue

                # Construct the command
                command = f"insanely-fast-whisper --file-name '{filepath}' --model '{args.model}' --transcript-path '{transcript_output}' --batch-size '{args.batch_size}'"

                # Log verbose information
                logging.debug(f"Processing file: {filename}")
                logging.debug(f"Command: {command}")

                # Execute the command and log output
                print(f"Processing {filename}...")
                result = run_command_and_log(command, log_file_path)

                if result != 0:
                    logging.error(f"Failed to process file: {filename}")
                else:
                    logging.info(f"{filename} processed successfully. Log saved to {log_file_path}.")

                    # Move the processed files to the appropriate directories
                    base_filename = os.path.splitext(filename)[0]
                    txt_output = os.path.join(args.processed_txt_dir, f"{base_filename}.txt")
                    srt_output = os.path.join(args.processed_srt_dir, f"{base_filename}.srt")

                    if os.path.exists(f"{base_filename}.txt"):
                        os.rename(f"{base_filename}.txt", txt_output)
                    if os.path.exists(f"{base_filename}.srt"):
                        os.rename(f"{base_filename}.srt", srt_output)

        # Log only once when all files are already processed
        if not files_found:
            if not no_new_files_logged:
                logging.info("No new files found. Waiting for 1 minute...")
                no_new_files_logged = True
            time.sleep(60)
        else:
            no_new_files_logged = False

if __name__ == "__main__":
    main()