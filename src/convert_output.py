import argparse
import json
import os
import time
import logging
from dotenv import load_dotenv
from datetime import datetime

# Load .env file if it exists
load_dotenv()

# Default values from the .env file
DEFAULT_TRANSCRIPTS = os.getenv("DEFAULT_TRANSCRIPTS", "transcripts")
DEFAULT_LOGS = os.getenv("DEFAULT_LOGS", "logs")
DEFAULT_VERBOSE = os.getenv("VERBOSE", "true").lower() in ("true", "1", "yes")
CONVERT_OUTPUT_FORMATS = os.getenv("CONVERT_OUTPUT_FORMATS", "txt,srt").split(',')
CONVERT_CHECK_INTERVAL = int(os.getenv("CONVERT_CHECK_INTERVAL", 60))
PROCESSED_FILES_TRACKER = os.path.join(DEFAULT_LOGS, "processed_files.json")

# New directories for processed files
PROCESSED_TXT_DIR = os.getenv("PROCESSED_TXT_DIR", "transcripts-txt")
PROCESSED_SRT_DIR = os.getenv("PROCESSED_SRT_DIR", "transcripts-srt")

# Ensure the directories exist
os.makedirs(PROCESSED_TXT_DIR, exist_ok=True)
os.makedirs(PROCESSED_SRT_DIR, exist_ok=True)

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
        if seconds is None:
            return "00:00:00,000"
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)

        hours = whole_seconds // 3600
        minutes = (whole_seconds % 3600) // 60
        seconds = whole_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    @classmethod
    def format_chunk(cls, chunk, index):
        text = chunk['text']
        start, end = chunk.get('timestamp', [None, None])
        start_format, end_format = cls.format_seconds(start), cls.format_seconds(end)
        return f"{index}\n{start_format} --> {end_format}\n{text}\n\n"

def setup_logging(log_file_path, verbose):
    # Clear existing handlers
    logging.getLogger().handlers.clear()

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
                            logging.StreamHandler()
                        ])

def flush_logs():
    """Force flush all log handlers."""
    for handler in logging.getLogger().handlers:
        handler.flush()

def load_processed_files():
    if os.path.exists(PROCESSED_FILES_TRACKER):
        with open(PROCESSED_FILES_TRACKER, 'r') as file:
            return json.load(file)
    return {}

def save_processed_file(filename, status="processed"):
    processed_files = load_processed_files()
    processed_files[filename] = {
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    with open(PROCESSED_FILES_TRACKER, 'w') as file:
        json.dump(processed_files, file, indent=4)

def main():
    transcripts_dir = DEFAULT_TRANSCRIPTS

    while True:
        processed_files = load_processed_files()
        files_found = False
        for filename in os.listdir(transcripts_dir):
            if filename.endswith(".json"):
                # Check if the file is already processed
                if filename in processed_files:
                    continue  # Skip this file if already processed

                files_found = True
                input_path = os.path.join(transcripts_dir, filename)

                # Set up logging for this specific file
                log_file_path = os.path.join(DEFAULT_LOGS, f"{os.path.splitext(filename)[0]}.log")
                setup_logging(log_file_path, DEFAULT_VERBOSE)

                try:
                    # Open and process the file
                    with open(input_path, 'r') as file:
                        data = json.load(file)

                    formatter_classes = {
                        'srt': SrtFormatter,
                        'txt': TxtFormatter
                    }

                    for output_format in CONVERT_OUTPUT_FORMATS:
                        formatter_class = formatter_classes.get(output_format)
                        string = formatter_class.preamble()

                        for index, chunk in enumerate(data['chunks'], 1):
                            entry = formatter_class.format_chunk(chunk, index)

                            if DEFAULT_VERBOSE:
                                print(entry)

                            string += entry

                        # Determine the output directory based on format
                        if output_format == 'txt':
                            output_dir = PROCESSED_TXT_DIR
                        elif output_format == 'srt':
                            output_dir = PROCESSED_SRT_DIR
                        else:
                            output_dir = transcripts_dir

                        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.{output_format}")
                        with open(output_file, 'w', encoding='utf-8') as output:
                            output.write(string)
                        logging.info(f"Successfully converted {input_path} to {output_file}")
                        flush_logs()

                    # Mark the file as processed
                    save_processed_file(filename, status="processed")

                except Exception as e:
                    logging.error(f"Error during conversion of {filename}: {str(e)}")
                    save_processed_file(filename, status="error")
                    flush_logs()

        if not files_found:
            time.sleep(CONVERT_CHECK_INTERVAL)
        else:
            time.sleep(CONVERT_CHECK_INTERVAL)

if __name__ == "__main__":
    main()