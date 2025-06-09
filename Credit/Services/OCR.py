import os
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from groq import Groq
import argparse
from dotenv import load_dotenv  # Add this import

def split_pdf_to_images(pdf_path, output_folder, image_format="PNG"):
    """
    Split a PDF file into separate image files, one for each page.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_folder (str): Folder to save the output image files
        image_format (str): Format to save images as (PNG, JPEG, etc.)
        
    Returns:
        list: Paths to the created image files
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Process each page
    image_files = []
    for page_num, page in enumerate(pdf_document):
        # Get the page as a pixmap (image)
        pix = page.get_pixmap(alpha=False)
        
        # Convert pixmap to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        # Set output image path
        output_image_path = os.path.join(output_folder, f"page_{page_num+1}.{image_format.lower()}")
        
        # Save as image
        img.save(output_image_path, image_format)
        
        image_files.append(output_image_path)
    
    pdf_document.close()
    
    return image_files

def encode_image_to_base64(image_path):
    """
    Convert an image file to base64 encoded string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def perform_ocr_with_llama_scout(image_path, client):
    """
    Perform OCR on an image using Llama-4-Scout model via GROQ SDK.
    
    Args:
        image_path (str): Path to the image file
        client: Groq client instance
        
    Returns:
        tuple: (extracted_text, error_info) - Text from image and error info if any
    """
    # Encode the image to base64
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        return "", {"error_type": "encoding_error", "message": str(e)}
    
    # Create the API request
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract all text from this image using OCR. Return only the extracted text, preserve all formatting, format everything in markdown format, and don't add any comments or explanations.

Instructions for standardizing data:
- If any information is missing, unclear or not found, indicate with "NA".
- For time periods:
  - Convert abbreviations like "Yrs", "Mths", "Days" to their full forms: "Years", "Months", "Days".
- For monetary values:
  - Convert Indian terms like "Lakhs", "Crores", and "Thousand" to full numeric values.
  - Example: "2.5 Lakhs" → 250000, "1,25,000" → 125000, "2 Crores" → 20000000
  - Remove currency symbols and commas from numbers.
- For dates, standardize to DD/MM/YYYY format where possible.
- For tables, maintain alignment and structure using markdown table format."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=4096,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        return completion.choices[0].message.content.strip(), None
    except Exception as e:
        error_info = {"error_type": "api_error", "message": str(e)}
        print(f"Error processing {image_path}: {str(e)}")
        return f"ERROR: Could not process {image_path}.", error_info

def extract_text_from_images(image_files, client):
    """
    Process a list of image files with OCR and return combined text.
    
    Args:
        image_files (list): List of image file paths
        client: Groq client instance
        
    Returns:
        tuple: (combined_text, errors_by_page) - Combined text and dict of errors by page
    """
    all_text = []
    errors_by_page = {}
    
    for i, image_path in enumerate(image_files):
        # Extract text using OCR
        extracted_text, error = perform_ocr_with_llama_scout(image_path, client)
        
        if error:
            page_num = i + 1
            errors_by_page[page_num] = {
                "page": page_num,
                "image_path": image_path,
                "error_type": error["error_type"],
                "message": error["message"]
            }
        
        # Add page header and the extracted text
        page_text = f"\n\n--- PAGE {i+1} ---\n\n{extracted_text}"
        all_text.append(page_text)
    
    # Combine all extracted text
    return "\n".join(all_text), errors_by_page

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PDF to Text Conversion with OCR')
    parser.add_argument('--pdf', default="2.pdf", help='Path to the input PDF file')
    parser.add_argument('--output', default="extracted_text.txt", help='Path to the output text file')
    parser.add_argument('--image_folder', default="output", help='Folder to save intermediate image files')
    parser.add_argument('--image_format', default="PNG", help='Image format (PNG, JPEG, etc.)')
    args = parser.parse_args()
    
    # Get API key from environment variable with better error handling
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is not set")
        return
    
    # Initialize the Groq client
    client = Groq(api_key=api_key)
    
    # Convert PDF to images
    print(f"Converting PDF to images...")
    image_files = split_pdf_to_images(args.pdf, args.image_folder, args.image_format)
    
    # Process images with OCR
    print(f"Processing images with OCR...")
    ocr_text, errors_by_page = extract_text_from_images(image_files, client)
    
    # Save to output file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(ocr_text)
    
    # Print errors if any
    if errors_by_page:
        print("Errors encountered during OCR:")
        for page, error_info in errors_by_page.items():
            print(f"Page {page}: {error_info['message']}")
    
    print(f"Text extraction complete. Saved to {args.output}")

if __name__ == "__main__":
    main()