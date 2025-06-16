# To-Do: Refactor main.py API Layer into Modular Structure

This plan outlines the steps to refactor the `main.py` FastAPI application layer into a modular structure, focusing on separation of concerns within the API layer while preserving the existing well-designed core architecture.

## Overview

The current `main.py` file mixes FastAPI application setup, route definitions, middleware, and request handling logic. The codebase already has excellent modular architecture in the `core/` directory with proper abstractions for ASR pipelines, backends, and storage. This refactoring focuses specifically on the API layer to improve maintainability and testability without duplicating existing functionality.

## Current State Analysis

**Existing Well-Designed Components (DO NOT MODIFY):**
- `core/pipeline.py`: Sophisticated ASR pipeline with Template Method and Observer patterns
- `core/asr_backend.py`: Clean backend abstraction with HuggingFace implementation
- `core/storage.py`: Storage abstraction and implementations
- `core/errors.py`: Comprehensive error handling
- `utils.py`: File handling utilities

**Issues to Address in main.py:**
- Mixed concerns: app setup, routes, middleware, and business logic in one file
- Duplicate file handling logic in both route functions
- Duplicate response formatting logic
- Long route functions with multiple responsibilities
- Lack of dependency injection for ASR pipeline instances

## Tasks

- [x] **Analysis Phase:**
  - [x] Review current implementation and supporting modules
    - Path: `insanely_fast_whisper_api/main.py`
    - Action: Analyze API layer structure, identify separation opportunities
    - Analysis Results:
      - Current architecture already has excellent core modules that should be preserved
      - Main issues are in API layer: mixed concerns in main.py, duplicate logic in routes
      - Need to separate FastAPI app setup, route definitions, and request handling
      - Existing core/pipeline.py already implements sophisticated patterns (Template Method, Observer)
    - **Design Pattern Candidates:**  
      - **Dependency Injection**: For ASR pipeline instances in routes
      - **Factory Pattern**: For FastAPI application creation
      - **Strategy Pattern**: For response formatting (already partially implemented)
    - Accept Criteria: Clear understanding of API layer separation needs while preserving core architecture
    - Status: Completed

- [ ] **Implementation Phase:**
  - [x] Create API module structure
    - Path: `insanely_fast_whisper_api/api/`
    - Action: Create focused API modules without duplicating core functionality
    - **New Structure:**
      - `app.py`: FastAPI application factory and startup configuration
      - `routes.py`: Clean route definitions with dependency injection
      - `middleware.py`: Request timing and logging middleware
      - `dependencies.py`: FastAPI dependency injection providers
      - `responses.py`: Response formatting utilities
    - **Design Patterns Applied:**  
      - Factory Pattern in `app.py` for FastAPI application creation
      - Dependency Injection in `dependencies.py` for ASR pipeline management
      - Strategy Pattern in `responses.py` for different response formats
    - **Completed Changes:**
      - Created `api/__init__.py` with package exports
      - Created `api/app.py` with FastAPI application factory
      - Created `api/middleware.py` with request timing middleware
      - Created `api/dependencies.py` with dependency injection for ASR pipeline and file handler
      - Created `api/responses.py` with Strategy pattern for response formatting
      - Created `api/routes.py` with clean, focused route definitions using dependency injection
      - Enhanced `utils.py` with FileHandler class for centralized file operations
    - Status: Completed
  - [x] Refactor main.py to use new API structure
    - Path: `insanely_fast_whisper_api/main.py`
    - Action: Simplify to entry point that uses app factory
    - **Completed Changes:**
      - Removed all FastAPI app setup, middleware, and route definitions from main.py
      - Simplified main.py to use the app factory pattern with `create_app()`
      - Reduced main.py from 349 lines to just 20 lines
      - Preserved logging configuration for console output
      - All functionality now properly separated into focused API modules
    - Status: Completed
  - [x] Enhance existing utils.py with FileHandler class
    - Path: `insanely_fast_whisper_api/utils.py`
    - Action: Add FileHandler class to centralize file operations
    - Status: Completed

- [x] **Testing Phase:**
  - [x] Write unit tests for new API modules
    - Path: `tests/test_api_modules.py`
    - Action: Test app factory, dependencies, response formatting
    - **Completed Tests:**
      - `TestAppFactory`: Tests for FastAPI application factory pattern
      - `TestDependencies`: Tests for dependency injection providers (ASR pipeline, file handler)
      - `TestResponseFormatter`: Tests for Strategy pattern response formatting (JSON/text)
      - `TestFileHandler`: Tests for centralized file operations (validation, save, cleanup)
      - `TestMiddleware`: Tests for request timing middleware functionality
    - **Coverage Areas:**
      - App creation and configuration
      - Dependency injection with proper parameter passing
      - Response formatting strategies for different output types
      - File handling operations with error scenarios
      - Middleware integration and logging
    - Accept Criteria: All API layer functionality covered by tests
    - Status: Completed
  - [x] Integration tests for refactored routes
    - Path: `tests/test_api_integration.py`
    - Action: Ensure routes work correctly with dependency injection
    - **Completed Tests:**
      - `TestTranscriptionEndpoint`: Full endpoint testing with mocked dependencies
      - `TestTranslationEndpoint`: Translation endpoint with JSON/text responses
      - `TestFileHandling`: File upload, processing, and cleanup integration
      - `TestBackwardsCompatibility`: Ensures API maintains same interface
    - **Coverage Areas:**
      - End-to-end request/response flow
      - Dependency injection in real request context
      - File validation and error handling
      - Response format handling (JSON vs text)
      - Parameter validation and backwards compatibility
      - Error scenarios and cleanup behavior
    - **Test Results:** All 29 tests passing successfully
    - Accept Criteria: All endpoints function identically to before refactoring
    - Status: Completed

- [ ] **Documentation Phase:**
  - [ ] Update `README.md`, `project-overview.md`
    - Describe new API module structure and how it integrates with existing core modules
    - Accept Criteria: Clear documentation of API layer architecture

- [ ] **Review Phase:**
  - [ ] Validate new structure maintains existing functionality while improving maintainability

## Architectural Overview

- **Preserved Core Architecture**:
  - `core/pipeline.py`: Keep existing WhisperPipeline with Template Method and Observer patterns
  - `core/asr_backend.py`: Keep existing HuggingFaceBackend and configuration
  - `core/storage.py`: Keep existing storage abstractions
  - `core/errors.py`: Keep existing error handling

- **New API Module Structure**:
  - `api/app.py`: FastAPI application factory, startup events, configuration
  - `api/routes.py`: Clean route definitions focused on request/response handling
  - `api/middleware.py`: Extracted middleware for cross-cutting concerns
  - `api/dependencies.py`: Dependency injection for ASR pipeline instances
  - `api/responses.py`: Response formatting strategies

- **Enhanced Utilities**:
  - Enhanced `utils.py` with FileHandler class for centralized file operations

- **Design Patterns Applied**:
  - Factory Pattern in `api/app.py` for application creation
  - Dependency Injection in `api/dependencies.py` for pipeline management
  - Strategy Pattern in `api/responses.py` for response formatting

- **Integration**: New API modules will use existing core functionality through clean interfaces, with dependency injection managing ASR pipeline instances and response formatters handling different output formats.

## Integration Points

- **Core Dependencies**:
  - `core/pipeline.py` (WhisperPipeline) - used via dependency injection
  - `core/asr_backend.py` (HuggingFaceBackend, HuggingFaceBackendConfig) - configuration management
  - `utils.py` (enhanced with FileHandler) - file operations
  - `constants.py` - configuration constants

- **Recommendations**: 
  - Use existing core modules as-is through dependency injection
  - Enhance file handling without breaking existing interfaces
  - Maintain compatibility with WebUI and CLI modules

## Related Files

- `insanely_fast_whisper_api/main.py` (to be refactored)
- `insanely_fast_whisper_api/core/pipeline.py` (use existing)
- `insanely_fast_whisper_api/core/asr_backend.py` (use existing)
- `insanely_fast_whisper_api/utils.py` (enhance existing)
- `insanely_fast_whisper_api/constants.py` (use existing)

## Implementation Examples

### api/app.py
```python
from fastapi import FastAPI
from .routes import router
from .middleware import add_middleware
from ..download_hf_model import download_model_if_needed

def create_app() -> FastAPI:
    """Factory function to create FastAPI application."""
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
    )
    
    add_middleware(app)
    app.include_router(router)
    
    @app.on_event("startup")
    async def startup_event():
        download_model_if_needed(model_name=None, custom_logger=logger)
    
    return app
```

### api/dependencies.py
```python
from fastapi import Depends
from ..core.pipeline import WhisperPipeline
from ..core.asr_backend import HuggingFaceBackend, HuggingFaceBackendConfig

def get_asr_pipeline(
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    # ... other config parameters
) -> WhisperPipeline:
    """Dependency to provide configured ASR pipeline."""
    backend_config = HuggingFaceBackendConfig(
        model_name=model,
        device=device,
        # ... other config
    )
    backend = HuggingFaceBackend(config=backend_config)
    return WhisperPipeline(asr_backend=backend)
```

### api/routes.py
```python
from fastapi import APIRouter, File, Form, UploadFile, Depends
from .dependencies import get_asr_pipeline
from .responses import ResponseFormatter
from ..utils import FileHandler

router = APIRouter()

@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    response_format: str = Form("json"),
    asr_pipeline: WhisperPipeline = Depends(get_asr_pipeline),
    file_handler: FileHandler = Depends(get_file_handler)
):
    """Clean, focused transcription endpoint."""
    # Validate and save file
    file_handler.validate_audio_file(file)
    temp_filepath = file_handler.save_upload(file)
    
    try:
        # Process using existing pipeline
        result = asr_pipeline.process(
            audio_file_path=temp_filepath,
            # ... other parameters
        )
        
        # Format response
        return ResponseFormatter.format_transcription(result, response_format)
    finally:
        file_handler.cleanup(temp_filepath)
```

## Future Enhancements

- Request-scoped ASR pipeline caching for better performance
- Async file processing for large uploads
- Enhanced error handling middleware with structured error responses
- Integration with existing progress tracking system from core/pipeline.py
- API versioning support 