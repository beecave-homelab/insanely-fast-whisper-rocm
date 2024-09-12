import gradio as gr
import os
import time
import logging
from dotenv import load_dotenv
import subprocess
import json
from main import setup_logging, run_command_and_log, convert
import queue
import threading
from pathlib import Path
import zipfile

# Load environment variables
load_dotenv()

# Default values
UPLOADS = os.getenv("UPLOADS", "/app/uploads")
TRANSCRIPTS = os.getenv("TRANSCRIPTS", "/app/transcripts")
LOGS = os.getenv("LOGS", "/app/logs")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
DEFAULT_MODEL = os.getenv("MODEL", "openai/whisper-large-v3")
DEFAULT_TASK = os.getenv("DEFAULT_TASK", "transcribe")
PROCESSED_TXT_DIR = os.getenv("PROCESSED_TXT_DIR", "transcripts-txt")
PROCESSED_SRT_DIR = os.getenv("PROCESSED_SRT_DIR", "transcripts-srt")

log_queue = queue.Queue()
file_queue = queue.Queue()
log_file_path = os.path.join(LOGS, "transcription.log")

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_queue.put(log_entry)

def setup_logging_with_queue(console_output=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    queue_handler = QueueHandler()
    queue_handler.setFormatter(formatter)
    logger.addHandler(queue_handler)

    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

def process_file(file_path, model, task):
    filename = os.path.basename(file_path)
    transcript_output = os.path.join(TRANSCRIPTS, f"{os.path.splitext(filename)[0]}.json")

    logging.info(f"Processing file: {filename}")
    command = f"insanely-fast-whisper --file-name '{file_path}' --model '{model}' --task '{task}' --transcript-path '{transcript_output}' --batch-size '{BATCH_SIZE}'"
    logging.info(f"Running command: {command}")

    result = run_command_and_log(command, log_file_path)

    if result != 0:
        logging.error(f"Failed to process file: {filename}")
        return None
    
    logging.info(f"{filename} processed successfully.")

    try:
        txt_output = os.path.join(PROCESSED_TXT_DIR, f"{os.path.splitext(filename)[0]}.txt")
        srt_output = os.path.join(PROCESSED_SRT_DIR, f"{os.path.splitext(filename)[0]}.srt")
        convert(transcript_output, 'txt', PROCESSED_TXT_DIR, True)
        convert(transcript_output, 'srt', PROCESSED_SRT_DIR, True)
        logging.info(f"Converted {os.path.basename(transcript_output)} to TXT and SRT formats.")
        file_queue.put((transcript_output, txt_output, srt_output))
    except Exception as e:
        logging.error(f"Error converting {os.path.basename(transcript_output)}: {str(e)}")

    return transcript_output

def create_zip_file(file_list, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in file_list:
            zipf.write(file, os.path.basename(file))
    return zip_filename

def create_gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown("# Audio Transcription App")
        
        file_input = gr.File(label="Upload Audio Files", file_count="multiple")
        model_input = gr.Textbox(label="Model", value=DEFAULT_MODEL)
        task_input = gr.Dropdown(label="Task", choices=["transcribe", "translate"], value=DEFAULT_TASK)
        process_button = gr.Button("Process")

        log_output = gr.Textbox(label="Processing Logs", lines=10)
        
        with gr.Row():
            json_download = gr.File(label="Download JSON", file_count="multiple")
            srt_download = gr.File(label="Download SRT", file_count="multiple")
            txt_download = gr.File(label="Download TXT", file_count="multiple")

        with gr.Row():
            json_download_all = gr.DownloadButton("Download All JSON")
            srt_download_all = gr.DownloadButton("Download All SRT")
            txt_download_all = gr.DownloadButton("Download All TXT")

        def process_and_update(files, model, task):
            if not files:
                yield [], [], [], "No files uploaded.", None, None, None
                return

            json_outputs = []
            srt_outputs = []
            txt_outputs = []
            log_content = ""

            setup_logging_with_queue(True)

            def process_files():
                for file in files:
                    process_file(file.name, model, task)

            thread = threading.Thread(target=process_files)
            thread.start()

            while thread.is_alive() or not log_queue.empty() or not file_queue.empty():
                while not log_queue.empty():
                    log_entry = log_queue.get()
                    log_content += log_entry + "\n"

                while not file_queue.empty():
                    json_file, txt_file, srt_file = file_queue.get()
                    json_outputs.append(json_file)
                    txt_outputs.append(txt_file)
                    srt_outputs.append(srt_file)

                json_zip = create_zip_file(json_outputs, "all_json_transcripts.zip")
                srt_zip = create_zip_file(srt_outputs, "all_srt_transcripts.zip")
                txt_zip = create_zip_file(txt_outputs, "all_txt_transcripts.zip")

                yield json_outputs, srt_outputs, txt_outputs, log_content, json_zip, srt_zip, txt_zip
                time.sleep(0.1)

            log_content += "All files processed."
            json_zip = create_zip_file(json_outputs, "all_json_transcripts.zip")
            srt_zip = create_zip_file(srt_outputs, "all_srt_transcripts.zip")
            txt_zip = create_zip_file(txt_outputs, "all_txt_transcripts.zip")
            yield json_outputs, srt_outputs, txt_outputs, log_content, json_zip, srt_zip, txt_zip

        process_button.click(
            process_and_update,
            inputs=[file_input, model_input, task_input],
            outputs=[json_download, srt_download, txt_download, log_output, json_download_all, srt_download_all, txt_download_all]
        )

    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)