# pipx install insanely-fast-whisper --force --pip-args="--ignore-requires-python" soundfile keyboard
import gradio as gr
import speech_recognition as sr
import torch
import torchaudio
import numpy as np
import keyboard
import logging

import time
from transformers import pipeline, file_utils
from transformers.utils import is_flash_attn_2_available

from pydub import AudioSegment
import os
import soundfile as sf
import onnxruntime as ort 
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ort.set_default_logger_severity(3) # remove warning

# Load environment variables from .env file
load_dotenv()

# Set the TORCH_HOME environment variable from .env or use default
torch_home = os.getenv("TORCH_HOME", "models")
os.environ['TORCH_HOME'] = torch_home

# Load Silero VAD model and utilities
try:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    vad_iterator = VADIterator(model)
    logging.info("Silero VAD model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Silero VAD model: {e}")
    raise

input_hijack = {
    'state': False,
    'value': ["", ""]
}

last_output = ""
previous_state = "silence"
buffer = np.array([])

def is_silence(audio):
    global buffer
    sample_rate, frame_data = audio  # Separate the sample rate and frame data
    window_size_samples = 512  # Number of samples in a single audio chunk
    speech_chunks = 0  # Counter for chunks detected as speech

    try:
        # Convert frame_data to Tensor if it isn't already
        if not isinstance(frame_data, torch.Tensor):
            frame_data = torch.tensor(frame_data, dtype=torch.float32)

        # Resample audio if necessary, but in smaller chunks to prevent memory overload
        if sample_rate not in [8000, 16000]:
            resample_rate = 16000  # Choose 16000 as a standard resample rate
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_rate)

            # Resample in chunks to avoid memory issues
            frame_data = resampler(frame_data)
            sample_rate = resample_rate
            logging.info(f"Audio resampled to {sample_rate} Hz.")

        # Ensure buffer is initialized correctly
        if buffer.size == 0:  # If buffer is empty
            buffer = np.zeros((0, frame_data.shape[1])) if frame_data.ndim == 2 else np.zeros(0)

        for i in range(0, frame_data.shape[0], window_size_samples):
            chunk = frame_data[i: i + window_size_samples]
            if chunk.ndim == 1:
                chunk = chunk.unsqueeze(1)  # Convert chunk to 2D if it is 1D

            buffer = np.concatenate((buffer[-2 * window_size_samples:], chunk.numpy()), axis=0)  # Concatenate along the correct axis

            if buffer.shape[0] < 3 * window_size_samples:  # If less than 3 chunks in buffer, continue to next iteration
                continue

            buffer_tensor = torch.from_numpy(buffer).float()  # Convert buffer to a PyTorch tensor and then to float
            speech_prob = model(buffer_tensor, sample_rate).item()  # Calculate speech probability for the buffer
            if speech_prob > 0.9:
                speech_chunks += 1
                if speech_chunks >= 2:
                    logging.info("Speech detected.")
                    return False

        vad_iterator.reset_states()
    except Exception as e:
        logging.error(f"Error in is_silence function: {e}")
        return True

    logging.info("No speech detected; returning silence.")
    return True

# Load Whisper model pipeline
try:
    pipe = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v2", torch_dtype=torch.float16, device="cuda:0", model_kwargs={"use_flash_attention_2": is_flash_attn_2_available()})
    if not is_flash_attn_2_available():
        logging.warning("Flash Attention 2.0 is not available. Using bettertransformer instead.")
        pipe.model = pipe.model.to('cuda')
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Whisper model: {e}")
    raise

def insanely_fast_whisper(audio_data):
    global last_output
    try:
        start_time = time.time()
        
        output = pipe(
            audio_data,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Whisper transcription completed in {elapsed_time} seconds.")

        last_output = output['text']
        return output['text']
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""

directory = os.path.dirname(os.path.realpath(__file__))
filename_voice = os.path.join(directory, 'temp_voice.wav')

def do_stt(audio):
    try:
        transcription = ""
        r = sr.Recognizer()

        # Convert to AudioData, Obtain the frame data as a NumPy array
        audio_data = sr.AudioData(sample_rate=audio[0], frame_data=audio[1], sample_width=4)
        new_data = np.frombuffer(audio_data.frame_data, dtype=np.int16)
        
        if os.path.exists(filename_voice):
            old_data, old_sample_rate = sf.read(filename_voice, dtype='int16')
            combined_data = np.concatenate((old_data, new_data)) # Concatenate the old data with the new data
            sf.write(filename_voice, combined_data, audio_data.sample_rate)
        else:
            sf.write(filename_voice, new_data, audio_data.sample_rate)
        
        transcription = insanely_fast_whisper(filename_voice)
        return transcription
    except Exception as e:
        logging.error(f"Error in do_stt function: {e}")
        return ""

def generate_transcribe():
    keyboard.send("enter")

def auto_transcribe(file_path, auto_submit):
    global last_output, previous_state
    try:
        audio, sample_rate = sf.read(file_path)
        audio = (sample_rate, audio)

        if not is_silence(audio):
            transcription = do_stt(audio)
            previous_state = "talking"
        else:
            transcription = ""
            if previous_state == "talking":
                generate_transcribe()
                transcription = last_output
            last_output = ""
            previous_state = "silence"
            sf.write(filename_voice, audio[1], audio[0])
        if auto_submit:
            input_hijack.update({"state": True, "value": [transcription, transcription]})
        return transcription, None
    except Exception as e:
        logging.error(f"Error in auto_transcribe function: {e}")
        return "", None

def ui():
    with gr.Blocks() as demo:
        with gr.Row():
            upload_button = gr.UploadButton(label="Upload Audio File", file_types=["audio"], file_count="single")
            auto_submit = gr.Checkbox(label='Submit the transcribed audio automatically', value=True, visible=False)
            text_box = gr.Textbox(label="Transcription Output")

        upload_button.upload(
            auto_transcribe, [upload_button, auto_submit], [text_box]
        ).then(
            None, auto_submit, None, js="(False) => { console.log('Check:', check); if (check) { document.getElementById('Generate').click(); }}"
        )
        
        demo.launch()

# Launch the Gradio interface
ui()