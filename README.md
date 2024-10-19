# Insanely Fast Whisper ROCm

This project is designed to provide a high-performance, GPU-accelerated environment for generating transcripts from audio files using AMD hardware and the ROCm platform. The project includes multiple scripts for different use cases, including an automatic file monitoring service (`main.py`) and a web-based user interface (`app.py`) built with Gradio. The setup is containerized using Docker and Docker Compose, ensuring a consistent and isolated environment optimized for ROCm.

## Table of Contents
- [Badges](#badges)
- [Installation](#installation)
- [Usage](#usage)
  - [Automatic Uploading Service](#automatic-uploading-service)
  - [Gradio Web UI](#gradio-web-ui)
- [License](#license)
- [Contributing](#contributing)

## Badges
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/docker-blue)
![ROCm](https://img.shields.io/badge/ROCm-6.1.2-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Installation

### Prerequisites
- **Docker**: Ensure that Docker (version 20.10 or newer) is installed and running on your system.
- **Docker Compose**: Ensure that Docker Compose is installed (comes bundled with Docker Desktop on Windows and Mac, or can be installed separately on Linux).
- **ROCm**: This project requires an AMD GPU that is compatible with ROCm 6.1.2.

### Steps
1. **Clone the repository**:
    ```bash
    git clone https://github.com/beecave-homelab/insanely-fast-whisper-rocm.git
    cd insanely-fast-whisper-rocm
    ```
2. **Create a `.env` file**:
    - Create a `.env` file in the root directory of the project with the necessary configuration. Example:
    ```bash
    UPLOADS_DIR=/app/uploads
    TRANSCRIPTS_DIR=/app/transcripts
    LOGS_DIR=/app/logs
    BATCH_SIZE=16
    VERBOSE=false
    ```

3. **Build the Docker image**:
    ```bash
    docker-compose build
    ```

4. **Run the Docker container**:
    ```bash
    docker-compose up -d
    ```

## Usage

### Automatic Uploading Service
The `main.py` script monitors a specified directory for new files and automatically generates transcripts. Follow these steps to use this feature:

1. **Start the service**:
    - Ensure the Docker container is running (`docker-compose up -d`).
    
2. **Place files in the `/uploads` directory**:
    - Any files added to this directory will be automatically processed, and the transcripts will be placed in the `/transcripts` directory.

3. **Check logs**:
    - Logs for the processing will be stored in the `/logs` directory.

### Gradio Web UI
The `app.py` script provides a web interface for uploading files and generating transcripts.

1. **Access the web interface**:
    - Navigate to `http://localhost:7860` in your web browser.
    
2. **Upload an audio file**:
    - Use the provided interface to upload an audio file. The file will be processed, and the transcript will be generated and displayed in the interface.

3. **View logs**:
    - Real-time logs are displayed in the web interface, and you can also find them in the `/logs` directory.

## License
This project is licensed under the MIT license. See [LICENSE](LICENSE) for more information.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.