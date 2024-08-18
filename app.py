import gradio as gr
import os
import time
import shutil
import threading
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
UPLOADS = os.environ.get("UPLOADS", "/app/uploads")
TRANSCRIPTS = os.environ.get("TRANSCRIPTS", "/app/transcripts")
LOGS = os.environ.get("LOGS", "/app/logs")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
DEFAULT_VERBOSE = os.environ.get("VERBOSE", "true").lower() in ("true", "1", "yes")

def setup_logging(verbose, file_path):
    """Configure logging."""
    # Generate the log file path based on the transcript output file name
    log_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}.json"
    log_file_path = os.path.join(LOGS, f"{os.path.splitext(log_file_name)[0]}.log")

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_path),
                            logging.StreamHandler()
                        ])

def validate_directories():
    """Check if required directories exist."""
    for directory in [UPLOADS, TRANSCRIPTS, LOGS]:
        if not os.path.isdir(directory):
            logging.error(f"Directory not found: {directory}")
            return False
    return True

def process_file(file_path: str, batch_size: int):
    """Process audio to generate transcript using a shell command."""
    transcript_output = os.path.join(TRANSCRIPTS, f"{os.path.splitext(os.path.basename(file_path))[0]}.json")

    if os.path.isfile(transcript_output):
        logging.warning(f"Transcript already exists: {transcript_output}")
        return transcript_output, ""

    # Construct the shell command
    cmd = f"insanely-fast-whisper --file-name '{file_path}' --transcript-path '{transcript_output}' --batch-size {batch_size}"

    def process():
        """Execute processing command in thread."""
        try:
            logging.info(f"Processing file: {file_path}")
            result = os.system(cmd)
            if result != 0:
                logging.error(f"Failed to process file: {file_path}")
            else:
                logging.info(f"Processing complete for: {file_path}")
        except Exception as e:
            logging.error(f"Error processing file: {file_path} - {str(e)}")

    threading.Thread(target=process).start()
    return transcript_output, ""

def create_gradio_interface():
    """Create Gradio app for file upload and processing."""
    with gr.Blocks() as app:
        gr.Markdown("# INSANELY-FAST-WHISPER | For AMD GPU (with rocm 6.1)")
        gr.Markdown("Upload an audio file that you want to process.")
        upload = gr.UploadButton(label="Upload Audio File", type="filepath")
        verbose_toggle = gr.Checkbox(label="Enable Verbose Logging", value=DEFAULT_VERBOSE)
        submit = gr.Button("Process File")

        with gr.Column():
            gr.Markdown("# Output Section")
            gr.Markdown("### Logs")
            gr.Markdown("Shows the processing logs.")
            logs_output = gr.Textbox(label="Logs", lines=10, max_lines=20, interactive=False)
            gr.Markdown("### Transcript")
            gr.Markdown("Displays the transcript of the uploaded audio file.")
            transcript_output = gr.Textbox(label="Transcript", lines=10, max_lines=20, interactive=False)

        def process_audio(file_object, verbose):
            if file_object is None:
                yield "No file uploaded.", ""
                return

            file_path = file_object.name
            setup_logging(verbose, file_path)  # Pass the file_path to setup_logging

            if not validate_directories():
                yield "Directory validation failed.", ""
                return

            full_file_path = os.path.join(UPLOADS, os.path.basename(file_path))
            shutil.move(file_path, full_file_path)  # Move file to UPLOADS directory

            output_paths = process_file(full_file_path, BATCH_SIZE)

            while True:
                try:
                    log_file_path = os.path.join(LOGS, f"{os.path.splitext(os.path.basename(full_file_path))[0]}.log")
                    with open(log_file_path, "r") as log_file:
                        logs_content = log_file.read()
                    yield logs_content, output_paths[0]
                except FileNotFoundError:
                    yield "Log file not found.", output_paths[0]

                time.sleep(5)  # Wait for 5 seconds before the next read

        submit.click(
            fn=process_audio,
            inputs=[upload, verbose_toggle],
            outputs=[logs_output, transcript_output],
            every=5  # This ensures that the function is called repeatedly every 5 seconds
        )

    return app

app = create_gradio_interface()

if __name__ == "__main__":
    app.launch()