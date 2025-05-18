# Insanely Fast Whisper ROCm

A high-performance speech-to-text transcription service using OpenAI's Whisper model optimized for AMD GPUs with ROCm support.

## Project Structure

```md
.
├── .dockerignore          # Docker ignore file
├── .env                   # Environment variables
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── Dockerfile            # Multi-stage Dockerfile for building the application
├── LICENSE               # Project license
├── README.md             # Project documentation
├── docker-compose.yaml    # Production Docker Compose configuration
├── docker-compose-dev.yaml # Development Docker Compose configuration
├── requirements*.txt     # Python dependencies
├── setup.sh              # Setup script
├── src/                  # Source code
│   ├── app.py           # Gradio web interface
│   ├── convert_output.py # Output format conversion utilities
│   └── main.py          # Core transcription functionality
├── logs/                 # Log files
├── testing/              # Test files
├── to-do/                # Task tracking
├── transcripts/          # Raw JSON transcriptions
├── transcripts-srt/      # SRT format transcriptions
├── transcripts-txt/      # Text format transcriptions
└── uploads/              # Uploaded audio/video files
```

## Development Workflow

This project uses Docker for development to ensure consistency across environments. All development and testing should be done within Docker containers.

### Prerequisites

- Docker and Docker Compose
- Git
- VS Code (recommended) or another IDE with Docker support

### Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/insanely-fast-whisper-rocm.git
   cd insanely-fast-whisper-rocm
   ```

2. **Set up environment variables**:

   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

3. **Build and start the development container**:

   ```bash
   docker-compose -f docker-compose-dev.yaml up --build
   ```

### Development Commands

- **Run the application**:

  ```bash
  docker-compose -f docker-compose-dev.yaml up
  ```

- **Run tests**:

  ```bash
  docker-compose -f docker-compose-dev.yaml run --rm app pytest
  ```

- **Access the container shell**:

  ```bash
  docker-compose -f docker-compose-dev.yaml exec app bash
  ```

- **View logs**:

  ```bash
  docker-compose -f docker-compose-dev.yaml logs -f
  ```

### VS Code Integration

1. Install the "Remote - Containers" extension
2. Open the project in VS Code
3. Click the green button in the bottom-left corner and select "Reopen in Container"
4. Wait for the container to build and start
5. Use the integrated terminal to run commands inside the container

### Debugging

- Set breakpoints in your code
- Use the VS Code debugger to attach to the running container
- View debug output in the Debug Console

## Key Components

### 1. Core Functionality (`main.py`)

- Handles the core transcription logic
- Implements file processing pipeline
- Manages logging and error handling
- Supports batch processing of audio files
- Includes formatters for different output formats (TXT, SRT, VTT)

### 2. Web Interface (`app.py`)

- Gradio-based web UI for file uploads
- Real-time progress tracking
- Support for batch processing
- File management and download functionality
- Log viewer

### 3. Output Conversion (`convert_output.py`)

- Converts JSON transcriptions to other formats
- Supports multiple output formats (TXT, SRT)
- Tracks processed files to avoid reprocessing
- Configurable output directories

## Containerization

The project uses Docker and Docker Compose for containerization with two main configurations:

### Production (`docker-compose.yaml`)

- Runs the Gradio web interface
- Maps port 7862 to container's 7860
- Includes volume mounts for persistent storage
- Uses the `main` tagged image

### Development (`docker-compose-dev.yaml`)

- Similar to production but with development settings
- Maps port 7863 to container's 7860
- Uses the `dev` tagged image
- Includes source code volume for live development

## Configuration

Environment variables can be configured in the `.env` file:

```env
# Core settings
UPLOADS="uploads"
TRANSCRIPTS="transcripts"
LOGS="logs"
BATCH_SIZE=6
VERBOSE=true
MODEL=distil-whisper/distil-large-v3

# Output conversion
CONVERT_OUTPUT_FORMATS="txt,srt"
CONVERT_CHECK_INTERVAL=120
PROCESSED_TXT_DIR="transcripts-txt"
PROCESSED_SRT_DIR="transcripts-srt"
```

## Dependencies

- Python 3.10+
- ROCm 6.1.2
- PyTorch with ROCm support
- Gradio for web interface
- Whisper model (default: distil-whisper/distil-large-v3)

## Usage

1. Configure environment variables in `.env`
2. Build and start with Docker Compose:

   ```bash
   docker-compose up --build
   ```

3. Access the web interface at `http://localhost:7862`

## Development

For development, use the development compose file:

```bash
docker-compose -f docker-compose-dev.yaml up --build
```

The web interface will be available at `http://localhost:7863`

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.
