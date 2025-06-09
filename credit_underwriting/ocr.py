from dotenv import load_dotenv
load_dotenv()

import base64
import io
import csv
from together import Together
from PIL import Image
from typing import List, Dict
import json
import os

class OCRProcessor:
    def __init__(self):
        self.api_key = os.getenv('LLM_API')
        self.temp_dir = os.getenv('TEMP_DIR', "temp_images")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.client = Together(api_key=self.api_key)
   
        
        self.fieldnames = [
            'Serial No', 'Customer ID', 'Account Number', 'Saving Account', 'Loan Type',
            'Loan Amount', 'Employement Type', 'Name (As Per ID)', 'Name (On Card)',
            'Maiden Name', "Father's Name", "Mother's Maiden Name", 'Residence Address',
            'Landmark', 'City', 'State', 'Country', 'Pin Code', 'Mobile Number',
            'Phone Number', 'Email ID', 'Constitution', 'Community', 'Aadhaar POI',
            'Aadhaar Present Address', 'Aadhaar Permanent Address', 'Aadhaar Number',
            'Annual Income', 'Net Monthly Income', 'Bank Name 1', 'Account Number 1',
            'Bank Name 2', 'Account Number 2', 'Loan Purpose', 'Credit Score',
            'Credit Utilization Ratio', 'Monthly Fixed Expenses', 'DTI', 'Work Experience',
            'Education Level', 'Marital Status', 'Dependents', 'Insurance Coverage',
            'Insurance Interest', 'Loan Repayment History', 'Litigation History',
            'Declaration Date', 'Declaration Place', 'Owns A House', 'Debt', 'Job Type',
            'Reference 1 Name', 'Reference 1 Relationship', 'Reference 1 Address',
            'Reference 1 Pin', 'Reference 1 City', 'Reference 1 State', 'Reference 1 Country',
            'Reference 1 Mobile', 'Reference 1 Email', 'Reference 2 Name',
            'Reference 2 Relationship', 'Reference 2 Address', 'Reference 2 Pin',
            'Reference 2 City', 'Reference 2 State', 'Reference 2 Country',
            'Reference 2 Mobile', 'Reference 2 Email'
        ]

    def extract_text_from_image(self, base64_image: str) -> Dict:
        try:
            data_uri = f"data:image/jpeg;base64,{base64_image}"
            
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": """
                            Extract data from the loan application form and return as simple text, Make sure its simple text no bold, italic use.
                            Follow these rules:
                            1. Return each field on a new line as 'Field: Value'
                            2. Do not use any markdown formatting or special characters
                            3. If a field has no value recheck the value before set as empty, write 'Field: "" '
                            4. Use exact field names as given
                            5. Do not add any headers or sections
                            6. For boolean fields (Yes/No answers), always use 'Yes' or 'No' with capital first letter
                            7. Standardize these fields to exactly 'Yes' or 'No':
                                - Saving Account
                                - Aadhaar POI
                                - Aadhaar Present Address 
                                - Aadhaar Permanent Address
                                - Insurance Coverage
                                - Insurance Interest
                                - Owns A House
                            8. Validate and convert these specific fields:
                                - Loan Type must be: 'Personal', 'Home', 'Vehicle', 'Education', 'Business'
                                - Litigation History must be: 'Clear' or 'Civil Case'
                                - Loan Repayment History must be: 'Excellent', 'Good', 'Fair', 'Poor'
                            
                            Required fields:
                            'Serial No', 'Customer ID', 'Account Number', 'Saving Account', 'Loan Type',
                            'Loan Amount', 'Employement Type', 'Name (As Per ID)', 'Name (On Card)',
                            'Maiden Name', "Father's Name", "Mother's Maiden Name", 'Residence Address',
                            'Landmark', 'City', 'State', 'Country', 'Pin Code', 'Mobile Number',
                            'Phone Number', 'Email ID', 'Constitution', 'Community', 'Aadhaar POI',
                            'Aadhaar Present Address', 'Aadhaar Permanent Address', 'Aadhaar Number',
                            'Annual Income', 'Net Monthly Income', 'Bank Name 1', 'Account Number 1',
                            'Bank Name 2', 'Account Number 2', 'Loan Purpose', 'Credit Score',
                            'Credit Utilization Ratio', 'Monthly Fixed Expenses', 'DTI', 'Work Experience',
                            'Education Level', 'Marital Status', 'Dependents', 'Insurance Coverage',
                            'Insurance Interest', 'Loan Repayment History', 'Litigation History',
                            'Declaration Date', 'Declaration Place', 'Owns A House', 'Debt', 'Job Type',
                            'Reference 1 Name', 'Reference 1 Relationship', 'Reference 1 Address',
                            'Reference 1 Pin', 'Reference 1 City', 'Reference 1 State', 'Reference 1 Country',
                            'Reference 1 Mobile', 'Reference 1 Email', 'Reference 2 Name',
                            'Reference 2 Relationship', 'Reference 2 Address', 'Reference 2 Pin',
                            'Reference 2 City', 'Reference 2 State', 'Reference 2 Country',
                            'Reference 2 Mobile', 'Reference 2 Email'
                        """},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }],
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stream=False
            )

            print("Processing response stream...")
            
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    response_text = choice.message.content
                    print(f"Response text: {response_text}")
                    
                    if response_text:
                        # Initialize data dict with empty strings
                        data = {field: "" for field in self.fieldnames}
                        
                        # Parse response line by line
                        for line in response_text.split('\n'):
                            line = line.strip()
                            if ':' in line:
                                key, value = [x.strip() for x in line.split(':', 1)]
                                
                                # Clean up key and value
                                key = key.lstrip('-*+ ')
                                value = value.strip('" {}')
                                
                                # Map common variations
                                key_mapping = {
                                    'Serial No.': 'Serial No',
                                    'Fathers Name': "Father's Name",
                                    'Mothers Maiden Name': "Mother's Maiden Name",
                                    'Employment Type': 'Employement Type'
                                }
                                key = key_mapping.get(key, key)
                                
                                # Store value if key exists in fieldnames
                                if key in self.fieldnames:
                                    # Standardize boolean fields
                                    boolean_fields = {
                                        'Saving Account', 'Aadhaar POI', 'Aadhaar Present Address',
                                        'Aadhaar Permanent Address', 'Insurance Coverage',
                                        'Insurance Interest', 'Owns A House'
                                    }
                                    if key in boolean_fields:
                                        # Convert various forms of yes/no to standardized format
                                        value_lower = value.lower()
                                        if value_lower in ['yes', 'y', 'true', '1']:
                                            value = 'Yes'
                                        elif value_lower in ['no', 'n', 'false', '0']:
                                            value = 'No'
                                    data[key] = value
                        
                        return data

            print("No valid response content found")
            return {field: "" for field in self.fieldnames}

        except Exception as e:
            print(f"Error in API query: {str(e)}")
            return {field: "" for field in self.fieldnames}

    def process_images(self, base64_images: List[str]) -> tuple[str, List[Dict]]:

        results = []
        
        for idx, base64_image in enumerate(base64_images, start=1):
            try:
                # Get extracted data from LLM
                extracted_data = self.extract_text_from_image(base64_image)
                
                # Create a complete record with all fields
                record = {field: "" for field in self.fieldnames}
                
                # Update with extracted data
                for key, value in extracted_data.items():
                    if key in self.fieldnames:
                        record[key] = value
                
                # Add serial number
                record["Serial No"] = str(idx)
                
                results.append(record)
                
            except Exception as e:
                print(f"Error processing image {idx}: {str(e)}")
                error_record = {field: "" for field in self.fieldnames}
                error_record["Serial No"] = str(idx)
                results.append(error_record)

        # Write to CSV
        output_file = "extracted_data.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        return output_file, results