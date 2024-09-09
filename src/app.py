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

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_queue.put(log_entry)

def setup_logging_with_queue(log_file_path, console_output=True):
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
    log_file_path = os.path.join(LOGS, f"{os.path.splitext(filename)[0]}.log")

    setup_logging_with_queue(log_file_path, True)

    logging.info(f"Processing file: {filename}")
    command = f"insanely-fast-whisper --file-name '{file_path}' --model '{model}' --task '{task}' --transcript-path '{transcript_output}' --batch-size '{BATCH_SIZE}'"
    logging.info(f"Running command: {command}")

    result = run_command_and_log(command, log_file_path)

    if result != 0:
        logging.error(f"Failed to process file: {filename}")
        return None
    
    logging.info(f"{filename} processed successfully. Log saved to {log_file_path}.")

    try:
        convert(transcript_output, 'txt', PROCESSED_TXT_DIR, True)
        convert(transcript_output, 'srt', PROCESSED_SRT_DIR, True)
        logging.info(f"Converted {os.path.basename(transcript_output)} to TXT and SRT formats.")
    except Exception as e:
        logging.error(f"Error converting {os.path.basename(transcript_output)}: {str(e)}")

    return transcript_output

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
            srt_download = gr.File(label="Download SRT", file_count="multiple", height=150)
            txt_download = gr.File(label="Download TXT", file_count="multiple", height=150)

        def process_and_update(files, model, task):
            if not files:
                yield [], [], [], "No files uploaded."
                return

            json_outputs = []
            srt_outputs = []
            txt_outputs = []
            log_content = ""

            for file in files:
                file_path = file.name
                
                def process_file_thread():
                    nonlocal file_path
                    return process_file(file_path, model, task)

                thread = threading.Thread(target=process_file_thread)
                thread.start()

                while thread.is_alive() or not log_queue.empty():
                    try:
                        log_entry = log_queue.get(timeout=0.1)
                        log_content += log_entry + "\n"
                        yield json_outputs, srt_outputs, txt_outputs, log_content
                    except queue.Empty:
                        time.sleep(0.1)

                transcript_output = process_file_thread()

                if transcript_output:
                    json_outputs.append(transcript_output)
                    srt_outputs.append(os.path.join(PROCESSED_SRT_DIR, f"{os.path.splitext(os.path.basename(file_path))[0]}.srt"))
                    txt_outputs.append(os.path.join(PROCESSED_TXT_DIR, f"{os.path.splitext(os.path.basename(file_path))[0]}.txt"))
                    log_content += f"Processing complete for {os.path.basename(file_path)}. Files ready for download.\n"
                else:
                    log_content += f"Processing failed for {os.path.basename(file_path)}. Unable to generate output files.\n"

            log_content += "All files processed."
            yield json_outputs, srt_outputs, txt_outputs, log_content

        process_button.click(
            process_and_update,
            inputs=[file_input, model_input, task_input],
            outputs=[json_download, srt_download, txt_download, log_output]
        )

    return app

if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)