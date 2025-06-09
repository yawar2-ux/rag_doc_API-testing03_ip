import os
import logging
from .FieldConfig import TEMP_DIR, UPLOAD_DIR, OUTPUT_DIR, STANDARD_CSV_FILENAME

def cleanup_temp_files():
    """
    Clean up temporary files and uploaded content.
    Only cleans TEMP_DIR and UPLOAD_DIR, preserving the OUTPUT_DIR.
    """
    try:
        # Clean TEMP_DIR
        if os.path.exists(TEMP_DIR):
            for file in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logging.error(f"Error deleting {file_path}: {str(e)}")
        
        # Clean UPLOAD_DIR
        if os.path.exists(UPLOAD_DIR):
            for file in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logging.error(f"Error deleting {file_path}: {str(e)}")
                    
    except Exception as e:
        logging.error(f"Error during temp cleanup: {str(e)}")
        
def cleanup_output_files():
    """
    Clean up OUTPUT_DIR files except for the standard CSV file.
    This should be called only after API processing is completely finished.
    """
    try:
        # Clean OUTPUT_DIR except for STANDARD_CSV_FILENAME
        if os.path.exists(OUTPUT_DIR):
            for file in os.listdir(OUTPUT_DIR):
                if file != STANDARD_CSV_FILENAME:  # Skip the standard CSV file
                    file_path = os.path.join(OUTPUT_DIR, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logging.error(f"Error deleting {file_path}: {str(e)}")
                    
    except Exception as e:
        logging.error(f"Error during output cleanup: {str(e)}")