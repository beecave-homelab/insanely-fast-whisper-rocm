```markdown
# Insanely Fast Whisper ROCm

This repository provides an implementation of Whisper, a highly efficient Automatic Speech Recognition (ASR) model, optimized for use with ROCm (Radeon Open Compute). The project leverages GPU acceleration for high-speed audio processing, making it suitable for applications requiring fast and accurate transcription.

## Table of Contents
- [Badges](#badges)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)

## Badges
![Python Version](https://img.shields.io/badge/python-3.9-blue)
![Docker](https://img.shields.io/badge/docker-available-green)
![License](https://img.shields.io/github/license/beecave-homelab/insanely-fast-whisper-rocm)

## Installation

### Prerequisites
- ROCm 6.1 or later
- Docker

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/beecave-homelab/insanely-fast-whisper-rocm.git
   ```
2. Build the Docker image:
   ```bash
   docker build -t whisper-rocm .
   ```

3. Run the Docker container:
   ```bash
   docker run --rm -it whisper-rocm
   ```

## Usage

### Running the Application

The `app.py` script is a Gradio-based application that provides an interface for uploading audio files and transcribing them using the Whisper model with ROCm acceleration.

### Key Features

- **Directory Monitoring and Processing**: 
  - The script continuously monitors a specified upload directory for new audio files.
  - Upon detecting a new file, it processes the file to generate a transcript, logging the process in real-time.
  - If the transcript already exists, the processing is skipped to avoid redundancy.

- **Gradio Interface**:
  - The Gradio interface allows users to upload files via a web-based UI, toggle verbose logging, and start/stop continuous directory monitoring.
  - Users can view real-time logs and transcriptions directly in the browser.

### Commands

- To start the application and use the Gradio interface:
  ```bash
  python app.py
  ```

- To transcribe an uploaded audio file, simply upload it through the Gradio UI and click "Process File".

- To enable continuous directory monitoring, select the appropriate checkbox in the UI and click "Start Monitoring".

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE) for more information.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
```
