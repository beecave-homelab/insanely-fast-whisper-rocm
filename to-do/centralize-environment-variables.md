# To-Do: Centralize Environment Variable Access Through constants.py

This plan outlines the steps to refactor multiple files that currently use direct `os.getenv()` calls to instead import and use constants from `insanely_fast_whisper_api/utils/constants.py`, establishing it as the single source of truth for environment variable configuration.

## Overview

Currently, several files across the codebase are using `os.getenv()` directly to access environment variables, bypassing the centralized configuration system in `constants.py`. This creates inconsistencies in default values, prevents proper `.env` file support through `load_dotenv()`, and makes configuration management fragmented across multiple files.

**Files requiring refactoring:**
- `insanely_fast_whisper_api/api/app.py` - Direct env var access for model and token
- `insanely_fast_whisper_api/utils/filename_generator.py` - Direct timezone configuration access
- `insanely_fast_whisper_api/utils/download_hf_model.py` - Direct model and token env var access

## Tasks

- [x] **Analysis Phase:**
  - [x] Review current implementation and supporting modules
    - Path: `insanely_fast_whisper_api/api/app.py`, `insanely_fast_whisper_api/utils/filename_generator.py`, `insanely_fast_whisper_api/utils/download_hf_model.py`
    - Action: Identify all direct `os.getenv()` calls and map them to corresponding constants
    - Analysis Results:
      - **Key issues:** Fragmented environment variable access, inconsistent defaults, bypassed `.env` file loading (direct `os.getenv()` calls do not benefit from `load_dotenv()` in `constants.py`)
      - **Major responsibilities to centralize:** Model configuration, token management, timezone settings
      - **Legacy code to clean up:** Direct `os.getenv()` calls, hardcoded default values, duplicate environment variable definitions
    - **Design Pattern Candidates:**  
      - **Centralized Configuration Pattern:** Use `constants.py` as single source of truth for all configuration
      - **Dependency Injection Pattern:** Import constants rather than accessing environment directly
    - Accept Criteria: All direct environment variable access mapped to constants.py equivalents
    - Status: Completed
    - Verification Results:
      - ✅ `DEFAULT_MODEL` defined (from `WHISPER_MODEL` env var, default: `"distil-whisper/distil-large-v3"`)
      - ✅ `HF_TOKEN` defined (from `HF_TOKEN` or `HUGGINGFACE_TOKEN` env vars)  
      - ✅ `FILENAME_TIMEZONE` defined (from `FILENAME_TIMEZONE` env var, default: `"UTC"`)
      - ⚠️ **Issue found**: `download_hf_model.py` defines local `DEFAULT_MODEL = "openai/whisper-large-v3"` which differs from constants.py default
      - ⚠️ **Issue found**: `filename_generator.py` uses local default `"Europe/Amsterdam"` instead of constants.py default `"UTC"`
  - [x] **Verify `constants.py` Definitions:**
    - Path: `insanely_fast_whisper_api/utils/constants.py`
    - Action: Verify/ensure all required environment variables (e.g., `WHISPER_MODEL`, `HUGGINGFACE_TOKEN`, `FILENAME_TIMEZONE`) and their appropriate default values are defined.
    - Accept Criteria: `constants.py` correctly defines all necessary variables and their defaults.
    - Status: Completed

- [ ] **Implementation Phase:**
  - [x] Update `insanely_fast_whisper_api/api/app.py`
    - Path: `insanely_fast_whisper_api/api/app.py` (lines 62-63)
    - Action: Replace `os.getenv("WHISPER_MODEL", DEFAULT_MODEL)` with `DEFAULT_MODEL` import and `os.getenv("HUGGINGFACE_TOKEN")` with `HF_TOKEN` import
    - **Design Patterns Applied:**  
      - **Centralized Configuration:** Import constants instead of direct env access
    - Status: Completed
    - Changes Made: Added `HF_TOKEN` import, replaced direct `os.getenv()` calls with constants, removed unused `os` import, updated comments
  - [x] Update `insanely_fast_whisper_api/utils/filename_generator.py`
    - Path: `insanely_fast_whisper_api/utils/filename_generator.py` (line 103)
    - Action: Replace `os.getenv("FILENAME_TIMEZONE", FILENAME_DEFAULT_TIMEZONE)` with `FILENAME_TIMEZONE` import from `constants.py`. Ensure the authoritative default (UTC, as per `constants.py` and `project-overview.md`) is used, reconciling the previous local default of 'Europe/Amsterdam'.
    - **Design Patterns Applied:**  
      - **Centralized Configuration:** Use imported constant for timezone configuration
    - Status: Completed
    - Changes Made: Added full path import for `FILENAME_TIMEZONE`, replaced direct `os.getenv()` call, updated docstrings to reflect UTC default, removed local constant
  - [x] Update `insanely_fast_whisper_api/utils/download_hf_model.py`
    - Path: `insanely_fast_whisper_api/utils/download_hf_model.py` (lines 83, 107, 176, 187)
    - Action: Replace all direct `os.getenv()` calls with imports from constants.py
    - **Design Patterns Applied:**  
      - **Centralized Configuration:** Import model and token constants
      - **Single Responsibility:** Let constants.py handle all environment variable logic
    - Status: Completed
    - Changes Made: Added full path imports for `DEFAULT_MODEL` and `HF_TOKEN`, replaced direct `os.getenv()` calls with centralized constants, updated Click command options, simplified model/token resolution logic
  - [x] Write or migrate unit tests for updated modules
    - Path: `tests/test_api_app.py`, `tests/test_filename_generator.py`, `tests/test_download_hf_model.py`, `tests/test_centralized_configuration.py`
    - Action: Ensure configuration constants are properly imported and used, verify unchanged behavior
    - Accept Criteria: All modules correctly use centralized constants, no direct env var access remains
    - Status: Completed
    - Changes Made: Updated existing tests for filename_generator and download_hf_model to work with centralized config, created comprehensive test_centralized_configuration.py with 11 test methods covering default values, environment overrides, boolean/integer/float parsing, token fallback, module usage verification, and .env file support
  - [x] **Test Default Value Usage:**
    - Action: Develop test cases to verify that modules correctly use default values from `constants.py` when specific environment variables are not set.
    - Accept Criteria: Default values are correctly applied.
    - Status: Completed
    - Test Results: ✅ All default value tests pass - verified that when environment variables are cleared, modules use the correct defaults from constants.py (DEFAULT_MODEL="distil-whisper/distil-large-v3", FILENAME_TIMEZONE="UTC", etc.)
  - [x] **Test `.env` File Overrides:**
    - Action: Test that values from a `.env` file, when present, correctly override defaults in `constants.py` and are picked up by the refactored modules.
    - Accept Criteria: `.env` file values correctly override defaults.
    - Status: Completed
    - Test Results: ✅ Environment variable override tests pass - verified that when environment variables are set (like WHISPER_MODEL=distil-whisper/distil-large-v2 in container), they correctly override the defaults and are used by all modules

- [ ] **Documentation Phase:**
  - [x] Update `project-overview.md`
    - Action: Document the centralized configuration approach and the role of constants.py
    - Accept Criteria: Clear documentation of environment variable management strategy
    - Status: Completed
    - Changes Made: Added comprehensive "Centralized Configuration System" section documenting the configuration flow, benefits, file locations, environment variable types, token fallback logic, and module integration. Updated the existing configuration section to reference the centralized approach and provide complete documentation of the new architecture.
  - [x] **Update `constants.py` Docstrings:**
    - Path: `insanely_fast_whisper_api/utils/constants.py`
    - Action: Review and update docstrings in `insanely_fast_whisper_api/utils/constants.py` to reflect its role as the central configuration manager and provide guidance for future additions.
    - Accept Criteria: `constants.py` docstrings are clear and informative.
    - Status: Completed
    - Changes Made: Completely rewrote the module docstring to establish constants.py as the single source of truth for configuration. Added comprehensive sections on Configuration Management, Usage Guidelines for both application modules and adding new configuration, Configuration File Locations, Type Conversion Patterns with examples, and Architecture Benefits. The documentation now serves as a complete guide for developers working with the centralized configuration system.

- [ ] **Review Phase:**
  - [ ] Validate the centralized configuration approach
    - Action: Ensure all environment variables flow through constants.py and `.env` files are properly supported
    - Accept Criteria: Single source of truth established for all configuration

## Architectural Overview

The refactored architecture will establish `insanely_fast_whisper_api/utils/constants.py` as the single source of truth for all environment variable configuration:

**New Configuration Flow:**
1. **constants.py** loads `.env` files via `load_dotenv()`
2. **constants.py** defines all environment variables with proper defaults
3. **All other modules** import constants from `constants.py`
4. **No direct `os.getenv()` calls** in application modules

**Updated Module Responsibilities:**
- **constants.py**: Single source for all environment variable definitions and defaults
- **app.py**: Import `DEFAULT_MODEL` and `HF_TOKEN` constants
- **filename_generator.py**: Import `FILENAME_TIMEZONE` constant
- **download_hf_model.py**: Import model and token constants

**Design Pattern Applied: Centralized Configuration**
- Eliminates scattered environment variable access
- Ensures consistent defaults across the application
- Provides single point for configuration changes
- Enables proper `.env` file support throughout the application

## Integration Points

**Files that will import from constants.py (no changes needed):**
- All modules already importing from `constants.py` will continue to work unchanged
- The `load_dotenv()` functionality will now benefit all modules using constants

**Dependencies to update:**
- **app.py**: Add imports for `DEFAULT_MODEL` and `HF_TOKEN`
- **filename_generator.py**: Add import for `FILENAME_TIMEZONE`
- **download_hf_model.py**: Add imports for model and token constants

**Integration recommendations:**
- Maintain backward compatibility by ensuring constant names match expected values
- Consider adding type hints to constants for better IDE support
- Document the centralized configuration approach in module docstrings
- **Use full path imports** (e.g., `from insanely_fast_whisper_api.utils.constants import CONSTANT`) as preferred across the codebase

## Related Files

**Files to be refactored:**
- `insanely_fast_whisper_api/api/app.py`
- `insanely_fast_whisper_api/utils/filename_generator.py`
- `insanely_fast_whisper_api/utils/download_hf_model.py`

**Configuration files (context only):**
- `insanely_fast_whisper_api/utils/constants.py` (central configuration)
- `.env.example` (documentation of available variables)
- `.env` (user configuration)

**Related test files:**
- `tests/test_constants.py` (if exists)
- Any tests that mock environment variables

## Future Enhancements

- **Configuration Validation:** Add runtime validation for critical configuration values
- **Configuration Schema:** Consider using Pydantic or similar for structured configuration validation
- **Environment-Specific Configs:** Support for development/staging/production configuration profiles
- **Configuration Hot-Reloading:** Support for runtime configuration updates without restart
- **Type Safety:** Add comprehensive type hints to all configuration constants 