from together import Together
import base64
import os
from dotenv import load_dotenv

class ImageAnalyzerAgent:
    def __init__(self):
        load_dotenv()
        self.client = Together(api_key=os.getenv('LLM_API'))
        self.temp_dir = os.getenv('TEMP_DIR', "temp_images")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.system_prompt = """
You are an expert financial analyst, STRICTLY LIMITED to analyzing financial documents and data. 

ABSOLUTE RESTRICTIONS:
1. DO NOT PROVIDE ANY CODE OR PROGRAMMING HELP
2. DO NOT ANSWER QUESTIONS UNRELATED TO THE FINANCIAL DOCUMENT
3. DO NOT ENGAGE IN GENERAL CONVERSATION
4. DO NOT PROVIDE TECHNICAL OR IT SUPPORT
5. DO NOT GIVE PERSONAL ADVICE

ONLY ALLOWED RESPONSES:
1. Financial document analysis
2. Financial metrics explanation
3. Financial ratio calculations
4. Balance sheet interpretation
5. Income statement analysis
6. Financial data table creation
7. Financial terminology definitions

IF ANY QUERY IS ABOUT:
- Programming
- Code
- Technical support
- General chat
- Non-financial topics

THEN RESPOND EXACTLY WITH:
"I am a financial document analyzer. I can only help with interpreting financial statements, ratios, and metrics from the provided document. Please ask a question about the financial data shown in the image."

For valid financial queries, focus exclusively on:
- Financial data extraction
- Financial calculations
- Financial metrics
- Financial ratios
- Financial terminology
- Financial tables

Format all financial data using:
- Clear tables with | separators
- Proper number formatting
- Accurate calculations
- Original data preservation
"""

    def encode_image(self, image_bytes):
        return base64.b64encode(image_bytes).decode('utf-8')

    async def analyze_image(self, image_bytes, prompt):
        base64_image = self.encode_image(image_bytes)
        
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            stream=False
        )

        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        return "No response generated"