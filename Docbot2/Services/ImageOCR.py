from groq import Groq
import base64
import os

# Initialize Groq client for image OCR
# Ensure GROQ_API_KEY2 is set in your environment variables
groq_api_key_ocr = os.environ.get("GROQ_API_KEY2")
if not groq_api_key_ocr:
    print("Warning: GROQ_API_KEY2 environment variable not set. Image OCR will fail.")
    # You might want to raise an error or have a fallback mechanism
    # For now, we'll let it proceed, and Groq client init will fail if key is truly needed and missing.
ocr_client = Groq(api_key=groq_api_key_ocr)

# Function to encode the image
def encode_image(image_path: str) -> str:
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(image_path: str) -> str:
    """
    Generates a description for an image using Groq's multimodal model.

    Args:
        image_path: Path to the image file.

    Returns:
        A string containing the description of the image.
        Returns an error message if OCR client is not available or API call fails.
    """
    if not groq_api_key_ocr: # Check if API key was available during init
        return "Error: Image OCR client not available due to missing GROQ_API_KEY2."

    try:
        base64_image = encode_image(image_path)

        chat_completion = ocr_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail. What objects, scenes, or text are visible?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}", # Assuming JPEG, adjust if other types are common
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct", # Ensure this model is available and supports image input
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during image OCR for {image_path}: {e}")
        return f"Error generating description for image {os.path.basename(image_path)}."

