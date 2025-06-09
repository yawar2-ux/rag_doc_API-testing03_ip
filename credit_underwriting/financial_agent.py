import pandas as pd
import json
import requests
import time
import os
from typing import Dict, List
from groq import Groq
from credit_underwriting.utils import print_step_header, print_record_status, print_financial_summary, print_completion_summary

class FinancialAgent:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.analysis_prompt = """As a Senior Financial Risk Analyst, evaluate loan applications using Indian banking standards and INR-denominated metrics to determine financial creditworthiness.

INPUT DATA:
{record}

EVALUATION CRITERIA:

1. Income Assessment (Score 0-100)
   - Strong: Fixed Expenses < 40% of Net Monthly Income
   - Intermediate: 40-60%
   - Weak: > 60%

2. Savings & Credit (Score 0-100)
   - Credit Utilization: Strong < 30%, Intermediate 30-50%, Weak > 50%
   - Monthly Savings: Strong > 30%, Intermediate 15-30%, Weak < 15%

3. Professional Profile (Score 0-100)
   - Experience: Strong > 3 years, Intermediate 1-3 years, Weak < 1 year
   - Include employment type and education level assessment

4. Assets & Stability (Score 0-100)
   - Evaluate: House ownership, savings account, insurance coverage
   - Consider: City tier and cost of living impact

RATING GUIDELINES:
- STRONG: At least 3 categories > 80 points, none below 60
- INTERMEDIATE: Most categories > 60 points, maximum one below 60
- WEAK: Multiple categories below 60 or any category below 40

RESPOND ONLY WITH A VALID JSON OBJECT IN THIS FORMAT:
{{
    "detailed_analysis": "Your thorough analysis of the financial position",
    "key_strengths": ["strength point 1", "strength point 2"],
    "key_concerns": ["concern point 1", "concern point 2"],
    "rating": "STRONG/INTERMEDIATE/WEAK"
}}

NOTE: Rating must be exactly one of: STRONG, INTERMEDIATE, or WEAK."""

    def query_llm(self, prompt: str) -> Dict:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user", 
                        "content": prompt + "\n\nRemember to respond with ONLY the JSON object and no additional text."
                    }
                ],
                model="llama-3.3-70b-versatile",
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            
            print("\n\nFinancial Agent ==> ", response_text)

            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                try:
                    json_str = response_text[start_idx:end_idx]
                    parsed_json = json.loads(json_str)

                    required_fields = [
                        "detailed_analysis",
                        "key_strengths",
                        "key_concerns",
                        "rating"
                    ]

                    if all(field in parsed_json for field in required_fields):
                        if parsed_json["rating"] not in ["STRONG", "INTERMEDIATE", "WEAK"]:
                            parsed_json["rating"] = "NO RATING PROVIDED"
                        return parsed_json

                except json.JSONDecodeError:
                    print("Failed to parse JSON response")

            return self._create_fallback_response("Failed to get valid analysis")

        except Exception as e:
            print(f"Error in LLM query: {str(e)}")
            return self._create_fallback_response(str(e))

    def _create_fallback_response(self, error_msg: str) -> Dict:
        return {
            "detailed_analysis": f"Analysis failed: {error_msg}",
            "key_strengths": ["Analysis incomplete"],
            "key_concerns": ["Technical error occurred"],
            "rating": "NO RATING PROVIDED"
        }

    def format_record(self, record: Dict) -> str:
        # Normalize Aadhaar address fields to Yes/No
        if 'Aadhaar Present Address' in record:
            record['Aadhaar Present Address'] = 'Yes' if str(record['Aadhaar Present Address']).lower() == 'yes' else 'No'
        if 'Aadhaar Permanent Address' in record:
            record['Aadhaar Permanent Address'] = 'Yes' if str(record['Aadhaar Permanent Address']).lower() == 'yes' else 'No'
            
        important_fields = [
            "Annual Income",
            "Net Monthly Income",
            "Monthly Fixed Expenses",
            "Credit Utilization Ratio",
            "Work Experience",
            "Owns A House",
            "Saving Account",
            "Constitution",
            "City",
            "Marital Status",
            "Education Level",
            "Insurance Coverage"
        ]

        return "\n".join(f"{field}: {record[field]}" for field in important_fields if field in record)

    def analyze_record(self, record: Dict) -> Dict:
        record_str = self.format_record(record)
        prompt = self.analysis_prompt.format(record=record_str)
        return self.query_llm(prompt)

    def process_records(self, input_file: str) -> str:
        try:
            print_step_header("Financial Analysis")

            df = pd.read_csv(input_file)
            results = []

            for idx, row in enumerate(df.iterrows(), 1):
                record_dict = row[1].to_dict()
                print_record_status(idx, len(df), f"Customer {record_dict.get('Customer ID', 'Unknown')}")

                analysis = self.analyze_record(record_dict)
                results.append({
                    "Customer_ID": record_dict.get("Customer ID", "Unknown"),
                    **analysis
                })

                print_financial_summary(
                    analysis.get('rating', 'UNKNOWN'),
                    analysis.get('key_strengths', []),
                    analysis.get('key_concerns', [])
                )

                time.sleep(1)

            output_file = "financial_analysis_results.csv"
            results_df = pd.DataFrame(results)
            
            # Convert account numbers to string format
            if 'Account Number' in results_df.columns:
                results_df['Account Number'] = results_df['Account Number'].astype(str)
            
            # Ensure loan amounts are numeric but not in scientific notation
            if 'Loan Amount' in results_df.columns:
                results_df['Loan Amount'] = pd.to_numeric(results_df['Loan Amount'])
                
            results_df.to_csv(output_file, index=False, float_format='%.0f')

            print_completion_summary(output_file, len(results))
            return output_file

        except Exception as e:
            print(f"Error: {str(e)}")
            return None