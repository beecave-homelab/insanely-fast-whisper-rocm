import os
import time
import logging
from dotenv import load_dotenv
import argparse

# Load .env file if it exists
load_dotenv()

# Default values
DEFAULT_UPLOADS = os.getenv("UPLOADS", "/app/uploads")
DEFAULT_TRANSCRIPTS = os.getenv("TRANSCRIPTS", "/app/transcripts")
DEFAULT_LOGS = os.getenv("LOGS", "/app/logs")
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
DEFAULT_VERBOSE = os.getenv("VERBOSE", "true").lower() in ("true", "1", "yes")
DEFAULT_MODEL = os.getenv("MODEL", "openai/whisper-large-v3")

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

def main():
    parser = argparse.ArgumentParser(description='Monitors a directory for file changes and processes files.')
    parser.add_argument('-u', '--uploads', default=DEFAULT_UPLOADS, help='Directory to watch for file changes')
    parser.add_argument('-t', '--transcripts', default=DEFAULT_TRANSCRIPTS, help='Directory to save transcript files')
    parser.add_argument('-l', '--logs', default=DEFAULT_LOGS, help='Directory to save log files')
    parser.add_argument('-b', '--batch-size', default=DEFAULT_BATCH_SIZE, type=int, help='Batch size for processing')
    parser.add_argument('-v', '--verbose', action='store_true', default=DEFAULT_VERBOSE, help='Enable verbose logging')
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, help='Model to be used for processing')

    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Validate directories
    for directory in [args.uploads, args.transcripts, args.logs]:
        if not os.path.isdir(directory):
            error_exit(f"Directory not found: {directory}")

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

                # Execute the command
                print(f"Processing {filename}...")
                result = os.system(command)

                if result != 0:
                    logging.error(f"Failed to process file: {filename}")
                else:
                    logging.info(f"{filename} processed successfully. Log saved to {log_file_path}.")

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