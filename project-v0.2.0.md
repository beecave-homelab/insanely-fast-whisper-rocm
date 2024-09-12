<!-- docs for project-v0.2.0 -->

# PROJECT: insanely-fast-whisper-rocm-v0.2.0

Refactor Insanely Fast Whisper ROCm project for improved modularity, maintainability, and scalability.

## SUMMARY: 

The project restructures the codebase, separating concerns into modules, introducing configuration management, and improving error handling and logging.

## STEPS:
1. Create new directory structure
2. Implement configuration management
3. Refactor main functionality into separate modules
4. Update main entry points
5. Implement improved error handling and logging
6. Update `Dockerfile` and `docker-compose.yml`
7. Update `README.md` with new structure and usage instructions

## STRUCTURE:
```
src/
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   ├── transcription.py
│   ├── conversion.py
│   └── file_management.py
├── utils/
│   ├── __init__.py
│   ├── logging.py
│   └── helpers.py
├── web/
│   ├── __init__.py
│   └── app.py
├── __init__.py
├── main.py
└── convert_output.py
```

## DETAILED EXPLANATION:
1. `config/settings.py`: Centralized configuration management
2. `core/transcription.py`: Core transcription functionality
3. `core/conversion.py`: Handles conversion of transcripts to different formats
4. `core/file_management.py`: Manages file operations and monitoring
5. `utils/logging.py`: Improved logging setup and utilities
6. `utils/helpers.py`: Common helper functions
7. `web/app.py`: Gradio web interface
8. `main.py`: Entry point for automatic file processing
9. `convert_output.py`: Entry point for conversion script

## SETUP:
1. Update `Dockerfile` to copy the new structure:
```
    # ... (previous Dockerfile content)
    COPY --chown=rocm-user:rocm-user src /app/src
```
2. Update `docker-compose.yml` to use the new entry points:
```   
command: ["python", "-m", "src.main"]  # or "src.web.app" or "src.convert_output"
```
3. Update `requirements.txt` to include any new dependencies.

## TAKEAWAYS:
1. Improved modularity and separation of concerns
2. Centralized configuration management
3. Enhanced error handling and logging
4. Easier maintenance and future development
5. Consistent coding style across the project

## SUGGESTIONS:
1. Implement unit tests for each module
2. Add type hints for improved code readability
3. Consider using asyncio for improved performance in file monitoring
4. Implement a plugin system for easy extension of functionality
5. Add comprehensive documentation for each module and function
