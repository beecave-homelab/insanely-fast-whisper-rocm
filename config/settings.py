"""Application settings and configuration."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, Field, validator
from pydantic.types import DirectoryPath, FilePath


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    # Core settings
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    model_name: str = Field(
        "distil-whisper/distil-large-v3",
        env="MODEL",
        description="HuggingFace model name for Whisper",
    )
    device: str = Field(
        "cuda", env="DEVICE", description="Device to run the model on (cuda, cpu, etc.)"
    )

    # File paths
    base_dir: DirectoryPath = Field(Path(__file__).parent.parent, env="BASE_DIR")
    uploads_dir: DirectoryPath = Field("uploads", env="UPLOADS")
    transcripts_dir: DirectoryPath = Field("transcripts", env="TRANSCRIPTS")
    logs_dir: DirectoryPath = Field("logs", env="LOGS")
    processed_txt_dir: DirectoryPath = Field("transcripts-txt", env="PROCESSED_TXT_DIR")
    processed_srt_dir: DirectoryPath = Field("transcripts-srt", env="PROCESSED_SRT_DIR")

    # Processing settings
    batch_size: int = Field(6, env="BATCH_SIZE", gt=0)
    num_workers: int = Field(2, env="NUM_WORKERS", ge=1)
    max_file_size_mb: int = Field(1024, env="MAX_FILE_SIZE_MB", gt=0)
    allowed_extensions: List[str] = Field(
        [".mp3", ".wav", ".m4a", ".mp4", ".mkv", ".webm", ".mpga", ".mpeg"],
        description="Allowed file extensions for upload",
    )

    # Output settings
    convert_output_formats: List[str] = Field(
        ["txt", "srt"],
        env="CONVERT_OUTPUT_FORMATS",
        description="List of output formats to convert to (txt, srt, vtt)",
    )
    convert_check_interval: int = Field(
        120,
        env="CONVERT_CHECK_INTERVAL",
        ge=1,
        description="Interval in seconds to check for new files to convert",
    )

    # Web settings
    web_host: str = Field("0.0.0.0", env="WEB_HOST")
    web_port: int = Field(7860, env="WEB_PORT")
    web_concurrency: int = Field(1, env="WEB_CONCURRENCY")
    web_reload: bool = Field(False, env="WEB_RELOAD")

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    @validator(
        "uploads_dir",
        "transcripts_dir",
        "logs_dir",
        "processed_txt_dir",
        "processed_srt_dir",
        pre=True,
    )
    def resolve_relative_paths(cls, v: str, values: dict, **kwargs) -> Path:
        """Resolve relative paths against base directory."""
        path = Path(v)
        if not path.is_absolute() and "base_dir" in values:
            return values["base_dir"] / path
        return path

    @validator("convert_output_formats", pre=True)
    def parse_convert_output_formats(cls, v: str) -> List[str]:
        """Parse comma-separated string of formats into a list."""
        if isinstance(v, str):
            return [fmt.strip().lower() for fmt in v.split(",") if fmt.strip()]
        return v

    def ensure_dirs_exist(self) -> None:
        """Ensure all required directories exist."""
        for dir_field in [
            self.uploads_dir,
            self.transcripts_dir,
            self.logs_dir,
            self.processed_txt_dir,
            self.processed_srt_dir,
        ]:
            dir_field.mkdir(parents=True, exist_ok=True)


# Create settings instance
settings = Settings()

# Ensure directories exist when module is imported
settings.ensure_dirs_exist()
