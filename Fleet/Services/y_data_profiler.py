#!/usr/bin/env python3
"""
YData Profiling utility functions and configurations.
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

# Use consolidated temp directory structure
TEMP_BASE_DIR = Path("temp_fleet_api")
TEMP_BASE_DIR.mkdir(exist_ok=True)

def cleanup_temp_files():
    """Utility function for cleaning up all temporary files in the base directory."""
    for temp_file in TEMP_BASE_DIR.rglob("*"):
        if temp_file.is_file():
            try:
                temp_file.unlink()
            except OSError: # More specific exception
                pass

def cleanup_specific_service_files(service_name: str):
    """Clean up temporary files for a specific service."""
    service_dir = TEMP_BASE_DIR / service_name
    if service_dir.exists():
        for temp_file in service_dir.glob("*"):
            try:
                temp_file.unlink()
            except OSError: # More specific exception
                pass

@asynccontextmanager
async def lifespan_cleanup():
    """Context manager for cleaning up temporary files."""
    # Startup: Clean up any existing temporary files
    cleanup_temp_files()
    yield
    # Shutdown: Clean up temporary files
    cleanup_temp_files()