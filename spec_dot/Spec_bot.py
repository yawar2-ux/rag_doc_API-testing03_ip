from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import time
import tempfile
import uuid
import hashlib
from pathlib import Path
import pandas as pd
from groq import Groq
# from dotenv import load_dotenv
import docx
import PyPDF2
import csv
import datetime
import json
from typing import Optional, Dict, Any
import logging
import traceback
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("genai-infra-recommender")
load_dotenv()
# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not set")
    raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

# Create temp directory for uploads and results
UPLOAD_DIR = Path("./uploads")
RESULTS_DIR = Path("./results")
CACHE_DIR = Path("./cache")  # Directory for caching results

for directory in [UPLOAD_DIR, RESULTS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# Store job statuses in memory (in production, use a database)
job_status = {}

MODEL = "llama-3.3-70b-versatile"  # Using Llama 3.3 70B

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
        return text
    except Exception as e:
        error_msg = f"Error extracting text from PDF: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

def extract_text_from_docx(file_path):
    """Extract text from a Word document"""
    try:
        text = ""
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        error_msg = f"Error extracting text from Word document: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

def extract_text_from_csv(file_path):
    """Extract text from a CSV file"""
    try:
        text = ""
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                text += ",".join(row) + "\n"
        return text
    except Exception as e:
        error_msg = f"Error extracting text from CSV: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

def extract_document_content(file_path):
    """Extract content from the document based on its file extension"""
    try:
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"Extracting content from: {file_path}")
        
        if file_extension == '.pdf':
            return extract_text_from_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return extract_text_from_docx(file_path)
        elif file_extension == '.csv':
            return extract_text_from_csv(file_path)
        else:
            error_msg = f"Unsupported file format: {file_extension}. Supported formats: .pdf, .docx, .doc, .csv"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = f"Error extracting document content: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

def generate_cache_key(document_content, special_instructions):
    """Generate a unique cache key based on document content and special instructions"""
    try:
        content_to_hash = f"{document_content or ''}|{special_instructions or ''}"
        return hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.error(f"Error generating cache key: {str(e)}\n{traceback.format_exc()}")
        # Return a unique key to prevent cache hits during errors
        return f"error-{uuid.uuid4()}"

def check_cache(cache_key):
    """Check if results exist in cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            logger.info(f"Cache hit for key {cache_key}")
            return cache_data.get("csv_content")
        return None
    except Exception as e:
        logger.error(f"Error checking cache: {str(e)}\n{traceback.format_exc()}")
        return None  # On cache error, proceed without using cache

def save_to_cache(cache_key, csv_content):
    """Save results to cache"""
    try:
        cache_file = CACHE_DIR / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({"csv_content": csv_content}, f)
        logger.info(f"Saved to cache with key {cache_key}")
    except Exception as e:
        logger.error(f"Error writing to cache: {str(e)}\n{traceback.format_exc()}")
        # Continue without error since caching is non-critical

def validate_csv_structure(csv_data):
    """Validate the CSV structure to ensure it meets requirements"""
    try:
        lines = csv_data.strip().split('\n')
        required_sections = ["Type,DC,Dev,UAT,DR,Total", "Type,Prod,Dev,UAT,DR,Disk Type", "Type,Cores,RAM [GB],Disk,NIC,GPU", "Type,DC,DR"]
        section_found = [False] * len(required_sections)
        
        # Check if all required sections are present
        for line in lines:
            for i, section_header in enumerate(required_sections):
                if line.startswith(section_header):
                    section_found[i] = True
                    break
        
        # If any section is missing, log an error
        if not all(section_found):
            missing_sections = [required_sections[i] for i, found in enumerate(section_found) if not found]
            logger.error(f"CSV validation failed: Missing sections: {missing_sections}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating CSV structure: {str(e)}\n{traceback.format_exc()}")
        return False

def enforce_csv_structure(csv_data):
    """Enforce proper CSV structure and fix common issues"""
    try:
        # Ensure proper line endings
        csv_data = csv_data.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into lines
        lines = csv_data.strip().split('\n')
        
        # Remove any markdown code blocks
        if lines and '```' in lines[0]:
            lines = lines[1:]
        if lines and '```' in lines[-1]:
            lines = lines[:-1]
        
        # Ensure consistent delimiters (some models might output semicolons instead of commas)
        for i in range(len(lines)):
            if ';' in lines[i] and ',' not in lines[i]:
                lines[i] = lines[i].replace(';', ',')
        
        # Ensure proper spacing between sections
        sections = []
        current_section = []
        
        for line in lines:
            if line.startswith('Type,'):
                if current_section:
                    sections.append(current_section)
                    current_section = []
            current_section.append(line)
        
        if current_section:
            sections.append(current_section)
        
        # Rebuild with proper spacing
        result_lines = []
        for i, section in enumerate(sections):
            result_lines.extend(section)
            if i < len(sections) - 1:
                result_lines.extend(['', ''])
        
        return '\n'.join(result_lines)
    except Exception as e:
        logger.error(f"Error enforcing CSV structure: {str(e)}\n{traceback.format_exc()}")
        return csv_data  # Return original in case of error

def generate_infrastructure_recommendations(document_content, job_id, special_instructions=None):
    """Generate infrastructure recommendations using Groq LLM"""
    
    # Update job status
    job_status[job_id]["status"] = "processing"
    job_status[job_id]["progress"] = 10
    
    try:
        # Check cache first
        cache_key = generate_cache_key(document_content, special_instructions)
        cached_result = check_cache(cache_key)
        
        if cached_result:
            job_status[job_id]["progress"] = 90
            logger.info(f"Using cached recommendations for job {job_id}")
            return cached_result
        
        # Set up system message for more consistent outputs
        system_message = """You are a specialized DevOps engineer and infrastructure sizing expert for AI systems. 
Your task is to analyze the provided document or requirements and generate an infrastructure sizing recommendation in CSV format.
You must maintain consistency and precision in your recommendations. Do not introduce randomness.
You should use a deterministic approach when calculating infrastructure needs.

Follow these principles for sizing:
1. For nodes and resources, use consistent ratios based on workload characteristics
2. Maintain a 1:1 ratio for DR to match production environments
3. Apply consistent scaling ratios (Dev = 25-33% of Prod, UAT = 33-50% of Prod)
4. Always recommend redundancy for critical components
5. Never deviate from the required CSV format
6. Ensure all calculations are consistent and reproducible

The CSV format is non-negotiable and must follow the exact structure provided in the example.
"""
        
        # Set up prompt based on whether we have document content, special instructions, or both
        if document_content and special_instructions and special_instructions.strip():
            # Both document and special instructions
            user_prompt = f"""Based on the following document and special instructions, create a deterministic and consistent infrastructure sizing recommendation.

DOCUMENT CONTENT:
{document_content}

SPECIAL INSTRUCTIONS FROM USER:
{special_instructions.strip()}

These special instructions should guide your analysis and recommendations. Consider these requirements when determining the infrastructure sizing.

Analyze this document and provide a complete infrastructure recommendation in EXACTLY the same CSV format as the example I'll provide. 
Follow the structure, headers, and table arrangement PRECISELY.
"""
            logger.info(f"Including both document content and special instructions for job {job_id}")
        
        elif document_content:
            # Only document content
            user_prompt = f"""Based on the following document that describes an application or requirements, create a deterministic and consistent infrastructure sizing recommendation.

DOCUMENT CONTENT:
{document_content}

Analyze this document and provide a complete infrastructure recommendation in EXACTLY the same CSV format as the example I'll provide. 
Follow the structure, headers, and table arrangement PRECISELY.
"""
            logger.info(f"Using only document content for job {job_id}")
        
        elif special_instructions and special_instructions.strip():
            # Only special instructions
            user_prompt = f"""Based on the following requirements description, create a deterministic and consistent infrastructure sizing recommendation.

REQUIREMENTS DESCRIPTION:
{special_instructions.strip()}

Analyze these requirements and provide a complete infrastructure recommendation in EXACTLY the same CSV format as the example I'll provide. 
Make appropriate assumptions where details are lacking, but aim to provide comprehensive sizing that satisfies the described needs.
Follow the structure, headers, and table arrangement PRECISELY.
"""
            logger.info(f"Using only special instructions for job {job_id}")
        
        else:
            # Neither document content nor special instructions (shouldn't happen)
            error_msg = "No input provided. Please provide either a document or requirements description."
            logger.error(f"No input provided for job {job_id}")
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = error_msg
            return None
        
        # Add the example CSV format to the prompt
        user_prompt += """
Here's the exact format to follow:

```
Type,DC,Dev,UAT,DR,Total,OEM,Physical/Virtual,,,DC,DR
Master Nodes,3,1,1,3,8,Translab,V,,Non AI Nodes,448,272
Utility Nodes,2,1,1,2,6,Translab,V,,AI Nodes,768,576
Data Nodes,3,1,1,3,8,Translab,P,,,,
AI Nodes,2,1,1,2,6,Translab,"P for DC and DR, V for Dev and UAT",,,,
DB Nodes,2,1,1,2,6,Translab,V,,,,
Visualization Nodes,2,1,1,2,6,Translab,V,,,,
,14,6,6,14,40,,,,,,
,,,,,,,,,,,
Type,Prod,Dev,UAT,DR,Disk Type,,,,,,
Object Storage [TB],100,10,20,100,NLSAS,,,,,,
,,,,,,,,,,,
,,,,,,,,,,,
Type,Cores,RAM [GB],Disk,NIC,GPU,Total Quantity,Total Cores,Total RAM,OS Storage (TB) (SAS),Data Storage (SSD),
Master Nodes,8,64,OS (SAS): 1 TB,2x 10/25 GbE,NA,8,64,512,8,,
Utility Nodes,8,64,OS (SAS): 1 TB,2x 10/25 GbE,NA,4,32,256,4,,
Data Nodes,16,256,OS (SAS): 1 TB,2x 10/25 GbE,NA,8,128,2048,8,,
AI Nodes (DC and DR),64,2048,"OS (SSD): 1 TB, Data (NVMe): 8 TB",2x 100 GbE ROCE,4x H100,4,256,8192,4,32,
AI Nodes (UAT and Dev),32,1024,"OS (SSD): 1 TB, Data (NVMe): 2 TB",2x 100 GbE ROCE,2x H100,2,64,2048,2,4,
DB Nodes,16,256,"OS (SSD): 1 TB, Data (NVMe): 3 TB",2x 10/25 GbE,NA,6,96,1536,6,18,
Visualization Nodes,16,128,OS (SAS): 1 TB,2x 10/25 GbE,NA,6,96,768,6,,
,,,,,,,,,,,
Type,DC,DR,,,,,,,,,
10/25 GbE [96 ports],2,2,,,,,,,,,
100 GbE ROCE [24 ports],2,2,,,,,,,,,
Rack,4,4,,,,,,,,,
,,,,,,,,,,,
Load Balancers,2,2,,,,,,,,,
Rack,5,5,,,,,,,,,
```

You must:
1. Keep the exact same table headers in the same positions
2. Maintain the same ordering of sections
3. Include the same blank rows between sections
4. Keep the same columns with empty values where shown in the example
5. Include the summary information on the right side of the first table (Non AI Nodes, AI Nodes)

Do NOT include any explanations or text apart from the CSV content. The entire response should consist of only the CSV with exact formatting.
Ensure ALL calculations (totals, etc.) are mathematically correct and consistent with the individual values.
"""
        
        job_status[job_id]["progress"] = 20
        
        # Initialize Groq client
        try:
            client = Groq(api_key=GROQ_API_KEY)
        except Exception as e:
            error_msg = f"Error initializing Groq client: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = error_msg
            return None
        
        # Make the API call with temperature set to ZERO for consistency
        try:
            job_status[job_id]["progress"] = 30
            
            # Use system and user messages for more control
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Set to zero for deterministic output
                max_tokens=4000
            )
            
            job_status[job_id]["progress"] = 80
            
            # Extract the CSV content from the response
            complete_response = response.choices[0].message.content
            
            # Clean and fix the CSV structure
            fixed_csv = enforce_csv_structure(complete_response)
            
            # Validate the CSV structure
            if not validate_csv_structure(fixed_csv):
                # If validation fails, retry once with more explicit instructions
                logger.warning(f"CSV validation failed for job {job_id}, retrying with stricter instructions")
                
                # Add more explicit instructions for retry
                retry_prompt = user_prompt + "\n\nIMPORTANT: Your previous response had structural issues. Make sure to include ALL four required table sections with their exact headers, and use precisely the format shown in the example."
                
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": retry_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=4000
                )
                
                complete_response = response.choices[0].message.content
                fixed_csv = enforce_csv_structure(complete_response)
                
                # If still invalid, raise an error
                if not validate_csv_structure(fixed_csv):
                    error_msg = "Failed to generate a valid infrastructure recommendation. Please try again with more detailed requirements."
                    logger.error(f"CSV validation failed again for job {job_id}")
                    job_status[job_id]["status"] = "failed"
                    job_status[job_id]["error"] = error_msg
                    return None
            
            job_status[job_id]["progress"] = 90
            logger.info(f"Recommendations generated for job {job_id}")
            
            # Save to cache for future requests
            save_to_cache(cache_key, fixed_csv)
            
            return fixed_csv
        except Exception as e:
            error_msg = f"Error generating recommendations: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            job_status[job_id]["status"] = "failed"
            job_status[job_id]["error"] = error_msg
            return None
    except Exception as e:
        error_msg = f"Unexpected error in generate_infrastructure_recommendations: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = error_msg
        return None

def ensure_results_dir():
    """Ensure the results directory exists"""
    try:
        results_dir = Path("./results")
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        return results_dir
    except Exception as e:
        logger.error(f"Error creating results directory: {str(e)}\n{traceback.format_exc()}")
        raise

def process_csv_data(csv_data):
    """Process and clean the CSV data returned by the LLM"""
    try:
        # Strip any markdown code block syntax
        csv_data = csv_data.strip()
        if csv_data.startswith("```csv"):
            csv_data = csv_data[6:]
        if csv_data.startswith("```"):
            csv_data = csv_data[3:]
        if csv_data.endswith("```"):
            csv_data = csv_data[:-3]
        
        csv_data = csv_data.strip()
        
        return csv_data
    except Exception as e:
        logger.error(f"Error processing CSV data: {str(e)}\n{traceback.format_exc()}")
        return csv_data  # Return original data if processing fails

def enhance_csv_formatting(csv_data):
    """Enhance the CSV formatting for better display in Excel"""
    try:
        # We'll ensure consistent spacing in the CSV file
        lines = csv_data.strip().split('\n')
        
        # Find the start of each section
        section_starts = []
        for i, line in enumerate(lines):
            if line.startswith('Type,'):
                section_starts.append(i)
        
        # Process the CSV to ensure proper separation
        processed_lines = []
        
        # Process each line
        for i, line in enumerate(lines):
            # Add the line
            processed_lines.append(line)
            
            # If this is the end of a section (and not the last section), ensure proper spacing
            if section_starts and i in [section_start + section_size - 1 for section_start, section_size in 
                    [(section_starts[j], section_starts[j+1]-section_starts[j]-1) 
                     for j in range(len(section_starts)-1)]]:
                # Make sure we have exactly two blank lines after sections
                blank_count = 0
                current_idx = i + 1
                while current_idx < len(lines) and not lines[current_idx].strip():
                    blank_count += 1
                    current_idx += 1
                
                # Adjust blank lines to exactly two
                if blank_count < 2:
                    for _ in range(2 - blank_count):
                        processed_lines.append('')
                # If there are more than 2 blank lines, they'll be skipped
        
        return '\n'.join(processed_lines)
    except Exception as e:
        logger.error(f"Error enhancing CSV formatting: {str(e)}\n{traceback.format_exc()}")
        return csv_data  # Return original data if enhancement fails

def save_csv_recommendations(csv_data, job_id):
    """Save the CSV data to a file with the job ID"""
    try:
        # Ensure results directory exists
        ensure_results_dir()
        
        # Rest of function remains the same
        enhanced_csv = enhance_csv_formatting(csv_data)
        csv_filename = RESULTS_DIR / f"{job_id}.csv"
        
        with open(csv_filename, 'w', newline='') as file:
            file.write(enhanced_csv)
        logger.info(f"CSV recommendations saved to {csv_filename}")
        return csv_filename
    except Exception as e:
        error_msg = f"Error saving CSV recommendations: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

def save_styled_excel(csv_data, job_id):
    """Save the recommendations to a beautifully styled Excel file"""
    try:
        # Ensure results directory exists
        ensure_results_dir()
        
        # Create the file path
        excel_filename = RESULTS_DIR / f"{job_id}.xlsx"
        
        # Split the CSV into lines
        lines = csv_data.strip().split('\n')
        
        # Create a Pandas Excel writer using XlsxWriter
        writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
        workbook = writer.book
        worksheet = workbook.add_worksheet('GenAI Sizing')
        
        # Define formats for different parts of the Excel file
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'align': 'center',
            'valign': 'vcenter',
            'font_color': 'white',
            'bg_color': '#4472C4',  # Blue
            'border': 1
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'font_color': 'white',
            'bg_color': '#4472C4',  # Blue
            'border': 1
        })
        
        node_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'bg_color': '#D9E1F2'  # Light blue
        })
        
        summary_header_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'font_color': 'white',
            'bg_color': '#70AD47',  # Green
            'border': 1
        })
        
        summary_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'bg_color': '#E2EFDA'  # Light green
        })
        
        storage_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'bg_color': '#FFF2CC'  # Light yellow
        })
        
        specs_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'bg_color': '#FCE4D6'  # Light orange
        })
        
        network_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'bg_color': '#DDEBF7'  # Very light blue
        })
        
        # Add a title at the top
        worksheet.merge_range('A1:K1', 'GenAI Infrastructure Sizing Recommendations', title_format)
        worksheet.write(1, 0, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
        
        row_offset = 3  # Start data after title and timestamp
        
        # Find the section boundaries
        section_starts = []
        for i, line in enumerate(lines):
            if line.startswith('Type,'):
                section_starts.append(i)
        
        # Process and write each section with appropriate formatting
        current_row = row_offset
        section_formats = [node_format, storage_format, specs_format, network_format]
        
        for i, line_index in enumerate(range(len(lines))):
            line = lines[line_index]
            if not line.strip():
                current_row += 1
                continue
            
            # Determine which section we're in
            current_section = 0
            for j, start in enumerate(section_starts):
                if line_index >= start:
                    current_section = j
            
            cells = line.split(',')
            
            # Determine the format for this row and write cells
            if line.startswith('Type,'):
                # This is a header row - write all cells with header format
                for j, cell in enumerate(cells):
                    if cell.strip():  # Only write non-empty cells
                        worksheet.write(current_row, j, cell, header_format)
            elif current_section == 0 and line_index > section_starts[0] and 9 < len(cells) and cells[9].strip():
                # This is the summary data on the right side of the first table
                if cells[9] == "Non AI Nodes" or cells[9] == "AI Nodes":
                    # Header for summary
                    worksheet.write(current_row, 9, cells[9], summary_header_format)
                    worksheet.write(current_row, 10, cells[10], summary_header_format)
                    worksheet.write(current_row, 11, cells[11], summary_header_format) if len(cells) > 11 else None
                else:
                    # Values for summary
                    worksheet.write(current_row, 9, cells[9], summary_format)
                    worksheet.write(current_row, 10, cells[10], summary_format)
                    worksheet.write(current_row, 11, cells[11], summary_format) if len(cells) > 11 else None
                
                # Now write the main row data
                for j, cell in enumerate(cells[:9]):
                    worksheet.write(current_row, j, cell, section_formats[current_section] if j > 0 else header_format if cell == "Type" else section_formats[current_section])
            else:
                # Regular data rows
                for j, cell in enumerate(cells):
                    if cell.strip():  # Only write non-empty cells
                        worksheet.write(current_row, j, cell, section_formats[current_section] if j > 0 or cell != "Type" else header_format)
            
            current_row += 1
        
        # Set column widths
        col_widths = {
            0: 25,  # Type column
            1: 10,  # DC column
            2: 10,  # Dev column
            3: 10,  # UAT column
            4: 10,  # DR column
            5: 10,  # Total column
            6: 15,  # OEM column
            7: 25,  # Physical/Virtual column
            8: 5,   # Spacing column
            9: 15,  # Summary header column
            10: 10, # DC stats column
            11: 10  # DR stats column
        }
        
        for col, width in col_widths.items():
            worksheet.set_column(col, col, width)
        
        # Freeze panes at the first data row
        worksheet.freeze_panes(row_offset, 0)
        
        # Save the Excel file
        writer.close()
        logger.info(f"Styled Excel saved to {excel_filename}")
        return excel_filename
    except Exception as e:
        error_msg = f"Error creating styled Excel file: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return None

def validate_and_fix_calculations(csv_data):
    """Validate and fix calculations in the CSV data to ensure consistency"""
    try:
        lines = csv_data.strip().split('\n')
        modified_lines = lines.copy()
        
        # First, find all the sections
        section_starts = []
        for i, line in enumerate(lines):
            if line.startswith('Type,'):
                section_starts.append(i)
        
        if len(section_starts) >= 1:
            # Fix the first section (nodes count)
            section_end = section_starts[1] - 1 if len(section_starts) > 1 else len(lines)
            section_rows = lines[section_starts[0]+1:section_end]
            
            # Extract data for calculations
            node_data = []
            for row in section_rows:
                if not row.strip() or not row.split(',')[0].strip():
                    continue
                    
                cells = row.split(',')
                if len(cells) < 6:
                    continue
                    
                try:
                    node_type = cells[0].strip()
                    dc = int(cells[1]) if cells[1].strip().isdigit() else 0
                    dev = int(cells[2]) if cells[2].strip().isdigit() else 0
                    uat = int(cells[3]) if cells[3].strip().isdigit() else 0
                    dr = int(cells[4]) if cells[4].strip().isdigit() else 0
                    
                    # Calculate and fix the total
                    total = dc + dev + uat + dr
                    
                    # Update the row with corrected total
                    row_cells = row.split(',')
                    row_cells[5] = str(total)
                    modified_row = ','.join(row_cells)
                    
                    # Find the index of this row and update it
                    row_index = lines.index(row)
                    modified_lines[row_index] = modified_row
                    
                    node_data.append((node_type, dc, dev, uat, dr, total))
                except Exception as e:
                    logger.warning(f"Error processing row: {row}, {str(e)}")
                    continue
            
            # Calculate totals row if we have node data
            if node_data:
                total_dc = sum(node[1] for node in node_data)
                total_dev = sum(node[2] for node in node_data)
                total_uat = sum(node[3] for node in node_data)
                total_dr = sum(node[4] for node in node_data)
                total_total = sum(node[5] for node in node_data)
                
                # Find the totals row (empty first column, numbers in others)
                for i in range(section_starts[0]+1, section_end):
                    cells = lines[i].split(',')
                    if len(cells) >= 6 and not cells[0].strip() and any(c.strip().isdigit() for c in cells[1:6]):
                        # Update the totals row
                        cells[1] = str(total_dc)
                        cells[2] = str(total_dev)
                        cells[3] = str(total_uat)
                        cells[4] = str(total_dr)
                        cells[5] = str(total_total)
                        modified_lines[i] = ','.join(cells)
                        break
        
        return '\n'.join(modified_lines)
    except Exception as e:
        logger.error(f"Error validating calculations: {str(e)}\n{traceback.format_exc()}")
        return csv_data  # Return original if validation fails

def process_request(file_path, job_id, special_instructions=None):
    """Process a document or special instructions and generate infrastructure recommendations"""
    try:
        # Set default document_content to None (for text-only requests)
        document_content = None
        
        # Extract content from document if it exists
        if file_path:
            document_content = extract_document_content(file_path)
            
            # Truncate document content if it's extremely large
            if len(document_content) > 15000:
                logger.warning(f"Document content is very large ({len(document_content)} characters). Truncating to 15,000 characters.")
                document_content = document_content[:15000]
        
        # Generate recommendations with the provided inputs
        csv_data = generate_infrastructure_recommendations(document_content, job_id, special_instructions)
        if not csv_data:
            # If generate_infrastructure_recommendations returned None, job_status should already have error information
            return
        
        # Process the CSV data
        csv_data = process_csv_data(csv_data)
        
        # Perform a final validation of totals and calculations for consistency
        csv_data = validate_and_fix_calculations(csv_data)
        
        # Save both CSV and styled Excel
        csv_filename = save_csv_recommendations(csv_data, job_id)
        excel_filename = save_styled_excel(csv_data, job_id)
        
        # Update job status
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["progress"] = 100
        job_status[job_id]["csv_path"] = str(csv_filename)
        job_status[job_id]["excel_path"] = str(excel_filename) if excel_filename else None
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = error_msg

def check_cache_status():
    """Get information about the current cache status"""
    try:
        cache_files = list(CACHE_DIR.glob("*.json"))
        total_cache_entries = len(cache_files)
        cache_size_bytes = sum(os.path.getsize(f) for f in cache_files)
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        
        return {
            "status": "success",
            "cache_entries": total_cache_entries,
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_directory": str(CACHE_DIR)
        }
    except Exception as e:
        error_msg = f"Error checking cache status: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "error": error_msg
        }

def clear_cache():
    """Clear the recommendation cache"""
    try:
        cache_files = list(CACHE_DIR.glob("*.json"))
        for file in cache_files:
            os.remove(file)
        
        return {
            "status": "success", 
            "message": f"Cleared {len(cache_files)} cache entries"
        }
    except Exception as e:
        error_msg = f"Error clearing cache: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "error": error_msg
        }

def cleanup_job(job_id: str):
    """
    Clean up files and status for a specific job.
    """
    try:
        if job_id not in job_status:
            raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
        
        # Delete uploaded files if they exist
        if "file_path" in job_status[job_id] and job_status[job_id]["file_path"]:
            job_dir = UPLOAD_DIR / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir)
        
        # Delete result files
        csv_path = job_status[job_id].get("csv_path")
        if csv_path and os.path.exists(csv_path):
            os.remove(csv_path)
        
        excel_path = job_status[job_id].get("excel_path")
        if excel_path and os.path.exists(excel_path):
            os.remove(excel_path)
        
        # Remove from status tracking
        del job_status[job_id]
        
        return {"message": f"Job {job_id} cleaned up successfully"}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_msg = f"Error cleaning up job: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_msg)

