import os
import time
import logging
import argparse
import json
import subprocess
import sys
import traceback

from dotenv import load_dotenv

# Load .env file if it exists
try:
    load_dotenv()
except Exception as e:
    print(f"Error loading .env file: {str(e)}")
    sys.exit(1)

# Default values with error handling
def get_env(key, default):
    try:
        return os.getenv(key, default)
    except Exception as e:
        print(f"Error getting environment variable {key}: {str(e)}")
        return default

DEFAULT_UPLOADS = get_env("UPLOADS", "/app/uploads")
DEFAULT_TRANSCRIPTS = get_env("TRANSCRIPTS", "/app/transcripts")
DEFAULT_LOGS = get_env("LOGS", "/app/logs")
DEFAULT_BATCH_SIZE = int(get_env("BATCH_SIZE", "4"))
DEFAULT_VERBOSE = get_env("VERBOSE", "true").lower() in ("true", "1", "yes")
DEFAULT_MODEL = get_env("MODEL", "openai/whisper-large-v3")
DEFAULT_PROCESSED_TXT_DIR = get_env("PROCESSED_TXT_DIR", "transcripts-txt")
DEFAULT_PROCESSED_SRT_DIR = get_env("PROCESSED_SRT_DIR", "transcripts-srt")

class TxtFormatter:
    @classmethod
    def preamble(cls):
        return ""

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk.get('text', '')
        return f"{text}\n"

class SrtFormatter:
    @classmethod
    def preamble(cls):
        return ""

    @classmethod
    def format_seconds(cls, seconds):
        if seconds is None:
            return "00:00:00,000"
        try:
            whole_seconds = int(seconds)
            milliseconds = int((seconds - whole_seconds) * 1000)

            hours = whole_seconds // 3600
            minutes = (whole_seconds % 3600) // 60
            seconds = whole_seconds % 60

            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
        except (TypeError, ValueError):
            logging.warning(f"Invalid timestamp value: {seconds}")
            return "00:00:00,000"

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk.get('text', '')
        start, end = chunk.get('timestamp', [None, None])
        start_format, end_format = cls.format_seconds(start), cls.format_seconds(end)
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"

class VttFormatter:
    @classmethod
    def preamble(cls):
        return "WEBVTT\n\n"

    @classmethod
    def format_seconds(cls, seconds):
        if seconds is None:
            return "00:00:00.000"
        try:
            whole_seconds = int(seconds)
            milliseconds = int((seconds - whole_seconds) * 1000)

            hours = whole_seconds // 3600
            minutes = (whole_seconds % 3600) // 60
            seconds = whole_seconds % 60

            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        except (TypeError, ValueError):
            logging.warning(f"Invalid timestamp value: {seconds}")
            return "00:00:00.000"

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk.get('text', '')
        start, end = chunk.get('timestamp', [None, None])
        start_format, end_format = cls.format_seconds(start), cls.format_seconds(end)
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"

def convert(input_path, output_format, output_dir, verbose):
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)

        formatter_class = {
            'srt': SrtFormatter,
            'vtt': VttFormatter,
            'txt': TxtFormatter
        }.get(output_format)

        if not formatter_class:
            raise ValueError(f"Unsupported output format: {output_format}")

        string = formatter_class.preamble()
        for index, chunk in enumerate(data.get('chunks', []), 1):
            entry = formatter_class.format_chunk(chunk, index)

            if verbose:
                print(entry)

            string += entry

        output_filename = os.path.splitext(os.path.basename(input_path))[0] + f".{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(string)
        
        logging.info(f"Successfully converted {input_path} to {output_path}")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {input_path}")
    except IOError:
        logging.error(f"IO error when processing {input_path}")
    except Exception as e:
        logging.error(f"Error during conversion of {input_path}: {str(e)}")
        logging.debug(traceback.format_exc())

def setup_logging(log_file_path, verbose):
    try:
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file_path),
                                logging.StreamHandler()
                            ])
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        sys.exit(1)

def error_exit(message):
    logging.error(message)
    sys.exit(1)

def create_directories(directories):
    for directory in directories:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
        except OSError as e:
            logging.error(f"Error creating directory {directory}: {str(e)}")
            sys.exit(1)

def run_command_and_log(command, log_file_path):
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        with open(log_file_path, 'a') as log_file:
            for line in process.stdout:
                log_file.write(line)
                print(line, end='')
                sys.stdout.flush()
        
        process.wait()
        
        return process.returncode
    except subprocess.SubprocessError as e:
        logging.error(f"Subprocess error running command: {e}")
    except Exception as e:
        logging.error(f"Error running command: {e}")
        logging.debug(traceback.format_exc())
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

    try:
        create_directories([args.uploads, args.transcripts, args.logs, args.processed_txt_dir, args.processed_srt_dir])

        if args.convert:
            if not args.input_file:
                error_exit("Input file must be specified for conversion.")
            log_file_path = os.path.join(args.logs, f"{os.path.splitext(os.path.basename(args.input_file))[0]}.log")
            setup_logging(log_file_path, args.verbose)
            convert(args.input_file, args.output_format, args.output_dir, args.verbose)
            return

        processed_files = set()

        while True:
            try:
                files_found = False
                no_new_files_logged = False

                for filename in os.listdir(args.uploads):
                    filepath = os.path.join(args.uploads, filename)

                    if os.path.isfile(filepath):
                        files_found = True
                        transcript_output = os.path.join(args.transcripts, f"{os.path.splitext(filename)[0]}.json")
                        log_file_path = os.path.join(args.logs, f"{os.path.splitext(filename)[0]}.log")

                        setup_logging(log_file_path, args.verbose)

                        if os.path.exists(transcript_output):
                            continue

                        command = f"insanely-fast-whisper --file-name '{filepath}' --model '{args.model}' --transcript-path '{transcript_output}' --batch-size '{args.batch_size}'"

                        logging.debug(f"Processing file: {filename}")
                        logging.debug(f"Command: {command}")

                        print(f"Processing {filename}...")
                        result = run_command_and_log(command, log_file_path)

                        if result != 0:
                            logging.error(f"Failed to process file: {filename}")
                        else:
                            logging.info(f"{filename} processed successfully. Log saved to {log_file_path}.")

                            try:
                                convert(transcript_output, 'txt', args.processed_txt_dir, args.verbose)
                                convert(transcript_output, 'srt', args.processed_srt_dir, args.verbose)
                                processed_files.add(os.path.basename(transcript_output))
                                logging.info(f"Converted {os.path.basename(transcript_output)} to TXT and SRT formats.")
                            except Exception as e:
                                logging.error(f"Error converting {os.path.basename(transcript_output)}: {str(e)}")
                                logging.debug(traceback.format_exc())

                if not files_found:
                    if not no_new_files_logged:
                        logging.info("No new files found. Waiting for 1 minute...")
                        no_new_files_logged = True
                    time.sleep(60)
                else:
                    no_new_files_logged = False

            except Exception as e:
                logging.error(f"An unexpected error occurred: {str(e)}")
                logging.debug(traceback.format_exc())
                time.sleep(60)

    except KeyboardInterrupt:
        logging.info("Script terminated by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        logging.debug(traceback.format_exc())

if __name__ == "__main__":
    main()