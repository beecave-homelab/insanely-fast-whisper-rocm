import gradio as gr
import os
import threading
import logging
from dotenv import load_dotenv

load_dotenv()
UPLOADS = os.environ.get("UPLOADS", "/app/uploads")
TRANSCRIPTS = os.environ.get("TRANSCRIPTS", "/app/transcripts")
LOGS = os.environ.get("LOGS", "/app/logs")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
DEFAULT_VERBOSE = os.environ.get("VERBOSE", "true").lower() in ("true", "1", "yes")

def setup_logging(verbose):
    """
    Configures the logging settings for the application.

    Parameters:
    verbose (bool): If True, sets the logging level to DEBUG for more detailed output. 
                    If False, sets the logging level to INFO for standard output.

    The logging configuration includes:
    - Logging messages to a file named 'script.log' located in the directory specified by the LOGS environment variable.
    - Logging messages to the console (standard output).

    The log message format includes the timestamp, log level, and the message.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f"{LOGS}/script.log"),
                            logging.StreamHandler()
                        ])

def validate_directories():
    """
    Validates the existence of required directories.

    This function checks if the directories specified by the UPLOADS, TRANSCRIPTS, 
    and LOGS environment variables exist. If any of these directories do not exist, 
    an error message is logged and the function returns False. If all directories exist, 
    the function returns True.

    Returns:
    bool: True if all required directories exist, False otherwise.
    """
    for directory in [UPLOADS, TRANSCRIPTS, LOGS]:
        if not os.path.isdir(directory):
            logging.error(f"Directory not found: {directory}")
            return False
    return True

def process_file(file_path: str, batch_size: int):
    """
    Processes an audio file to generate a transcript.

    This function generates a transcript for the provided audio file by running an external command.
    If a transcript already exists for the given file, a warning is logged and the function returns
    the path to the existing transcript.

    Parameters:
    file_path (str): The path to the audio file to be processed.
    batch_size (int): The batch size to be used for processing.

    Returns:
    tuple: A tuple containing the path to the transcript file and an empty string.
           If the transcript already exists, the path to the existing transcript is returned.
    """
    transcript_path = os.path.join(TRANSCRIPTS, f"{os.path.basename(file_path)}.txt")
    if os.path.isfile(transcript_path):
        logging.warning(f"Transcript already exists: {transcript_path}")
        return transcript_path, ""

    cmd = f"python3 main.py --input '{file_path}' --output '{transcript_path}' --batch_size {batch_size}"

    def process():
        """
        Processes the given file by executing a system command.

        This function logs the start and completion of the file processing. 
        If an error occurs during the execution of the system command, 
        the error is logged.

        The processing is performed in a separate thread to avoid blocking 
        the main thread.

        Returns:
        str: The path to the transcript file.
        str: An empty string.
        """
        try:
            logging.info(f"Processing file: {file_path}")
            os.system(cmd)
            logging.info(f"Processing complete for: {file_path}")
        except Exception as e:
            logging.error(f"Error processing file: {file_path} - {str(e)}")

    threading.Thread(target=process).start()

    return transcript_path, ""

def monitor_directory(batch_size, verbose):
    """
    Monitors the upload directory for new files and processes them.

    This function continuously monitors the directory specified by the UPLOADS environment variable 
    for new audio files. When a new file is found, it processes the file using an external command 
    to generate a transcript. The function logs the start and completion of file processing, as well 
    as any errors encountered during processing. If no new files are found, it waits for 1 minute 
    before checking again.

    Parameters:
    batch_size (int): The batch size to be used for processing.
    verbose (bool): If True, sets the logging level to DEBUG for more detailed output. 
                    If False, sets the logging level to INFO for standard output.

    The function performs the following steps:
    1. Configures logging based on the verbose parameter.
    2. Validates the existence of required directories (UPLOADS, TRANSCRIPTS, LOGS).
       If validation fails, logs an error and exits.
    3. Logs the start of directory monitoring.
    4. Enters an infinite loop to monitor the UPLOADS directory.
       - Checks for new files in the directory.
       - If a new file is found, constructs the transcript output path and log file path.
       - If the transcript already exists, waits for 1 minute and continues to the next file.
       - Constructs and executes the command to process the file.
       - Logs the result of the processing.
       - If no new files are found, logs a message and waits for 1 minute before checking again.

    Note: The function runs indefinitely until manually stopped.

    Returns:
    None
    """
    setup_logging(verbose)
    
    logging.info("Starting directory monitoring...")
    
    while True:
        files_found = False
        no_new_files_logged = False

        for filename in os.listdir(UPLOADS):
            filepath = os.path.join(UPLOADS, filename)
            
            if os.path.isfile(filepath):
                files_found = True
                transcript_output = os.path.join(TRANSCRIPTS, f"{os.path.splitext(filename)[0]}.json")
                log_file_path = os.path.join(LOGS, f"{os.path.splitext(filename)[0]}.log")

                if os.path.exists(transcript_output):
                    time.sleep(60)
                    continue
                
                command = f"insanely-fast-whisper --file-name '{filepath}' --transcript-path '{transcript_output}' --batch-size '{batch_size}'"
                
                logging.debug(f"Processing file: {filename}")
                logging.debug(f"Command: {command}")

                print(f"Processing {filename}...")
                result = os.system(command)

                if result != 0:
                    logging.error(f"Failed to process file: {filename}")
                else:
                    logging.info(f"{filename} processed successfully. Log saved to {log_file_path}.")

        if not files_found:
            if not no_new_files_logged:
                logging.info("No new files found. Waiting for 1 minute...")
                no_new_files_logged = True
            time.sleep(60)
        else:
            no_new_files_logged = False

def create_gradio_interface():
    """
    Creates a Gradio interface for uploading and processing audio files for transcription.

    This function sets up a web interface using Gradio, allowing users to upload audio files,
    toggle verbose logging, and enable continuous monitoring of a directory for new files. 
    The interface has two main sections: an upload section and an output section.

    The upload section includes:
    - A button to upload audio files.
    - A checkbox to enable or disable verbose logging.
    - A checkbox to enable or disable continuous monitoring of the upload directory.
    - A button to manually process the uploaded file.
    - A button to start continuous monitoring.

    The output section includes:
    - A textbox to display logs from the processing.
    - A textbox to display the transcript of the uploaded audio file.

    The function defines two main operations:
    1. `process_audio`: Processes the uploaded audio file, logs the process, and displays the transcript.
    2. `start_monitoring`: Starts a background thread to continuously monitor the upload directory for new files.

    Returns:
    gr.Blocks: The Gradio interface application.
    """
    with gr.Blocks() as app:
                gr.Markdown(
                    """
                    Upload an audio file that you want to process.
                    """
                )
                upload = gr.UploadButton(label="Upload Audio File", type="filepath")
                verbose_toggle = gr.Checkbox(label="Enable Verbose Logging", value=DEFAULT_VERBOSE)
                continuous_monitoring = gr.Checkbox(label="Enable Continuous Monitoring")
                gr.Markdown(
                    """
                    After the file is uploaded, click 'Process File' to start transcription.
                    """
                )
                submit = gr.Button("Process File")
                monitor_button = gr.Button("Start Monitoring")

            with gr.Column():
                gr.Markdown("### Output Section")
                logs_output = gr.Textbox(label="Logs", lines=10, max_lines=20, interactive=False)
                gr.Markdown("Shows the processing logs.")
                transcript_output = gr.Textbox(label="Transcript", lines=10, max_lines=20, interactive=False)
                gr.Markdown("Displays the transcript of the uploaded audio file.")

        def process_audio(file_path, verbose):
            setup_logging(verbose)
            if not validate_directories():
                return "Directory validation failed.", ""
            
            file_path = file_path.name
            full_file_path = os.path.join(UPLOADS, file_path)
            output_paths = process_file(full_file_path, BATCH_SIZE)

            try:
                with open(os.path.join(LOGS, f"{os.path.basename(file_path)}.log"), "r") as log_file:
                    logs_content = log_file.read()
                return logs_content, output_paths[0]
            except FileNotFoundError:
                return "Log file not found.", output_paths[0]

        def start_monitoring(batch_size, verbose):
            threading.Thread(target=monitor_directory, args=(batch_size, verbose)).start()
            return "Monitoring started..."

        submit.click(
            fn=process_audio,
            inputs=[upload, verbose_toggle],
            outputs=[logs_output, transcript_output]
        )

        monitor_button.click(
            fn=start_monitoring,
            inputs=[gr.Number(value=BATCH_SIZE, label="Batch Size"), verbose_toggle],
            outputs=logs_output
        )

    return app

app = create_gradio_interface()

if __name__ == "__main__":
    app.launch()