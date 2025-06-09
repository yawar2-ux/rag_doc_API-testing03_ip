import pandas as pd
import json
import requests
import time
import os
from typing import Dict, List
from groq import Groq
from credit_underwriting.utils import print_step_header, print_record_status, print_risk_summary, print_completion_summary

class RiskAgent:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.analysis_prompt = """As a Senior Risk Assessment Specialist with expertise in loan underwriting, conduct comprehensive risk evaluation of loan applications using established risk assessment frameworks and Indian banking regulatory guidelines. Your analysis will determine the overall risk profile considering both quantitative and qualitative risk indicators.

PRIMARY OBJECTIVE:
Evaluate the applicant's risk profile through systematic assessment of credit history, income stability, loan parameters, and personal risk factors to determine lending risk exposure.

INPUT DATA:
{record}

RISK EVALUATION PARAMETERS:

1. Credit Profile Risk Assessment:
   - Strong: Credit Score > 750, DTI < 40%, Clean repayment history
   - Moderate: Credit Score 650-750, DTI 40-50%, Minor defaults
   - High Risk: Credit Score < 650, DTI > 50%, Multiple defaults

2. Income & Employment Risk:
   - Low Risk: Permanent role, > 3 years in stable sector
   - Moderate Risk: 1-3 years, Growing sector
   - High Risk: < 1 year, Volatile sector/Contract role

3. Loan Parameter Risk:
   - Low Risk: Purpose aligned with income, Amount < 3x annual income
   - Moderate Risk: Purpose justified, Amount 3-5x annual income
   - High Risk: Unclear purpose, Amount > 5x annual income

4. Personal Risk Factors:
   - Low Risk: Verified addresses match, Adequate insurance
   - Moderate Risk: Recent address change, Basic insurance
   - High Risk: Address mismatch, No insurance

RISK RATING DETERMINATION:
- LOW_RISK: All parameters in Low/Strong range, maximum one Moderate
- MEDIUM_RISK: Mix of ratings, maximum one High Risk factor
- HIGH_RISK: Multiple High Risk factors or severe risk in any category

RESPOND ONLY WITH A VALID JSON OBJECT IN THIS FORMAT:
{{
    "detailed_risk_analysis": "Your thorough analysis of all risk factors",
    "major_risk_factors": ["risk factor 1", "risk factor 2"],
    "mitigating_factors": ["mitigating factor 1", "mitigating factor 2"],
    "risk_rating": "HIGH_RISK/MEDIUM_RISK/LOW_RISK"
}}

NOTE: risk_rating must be exactly one of: HIGH_RISK, MEDIUM_RISK, or LOW_RISK."""

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
            
            print("\n\nRisk Agent ==> ", response_text)

            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                try:
                    json_str = response_text[start_idx:end_idx]
                    parsed_json = json.loads(json_str)

                    required_fields = [
                        "detailed_risk_analysis",
                        "major_risk_factors",
                        "mitigating_factors",
                        "risk_rating"
                    ]

                    if all(field in parsed_json for field in required_fields):
                        # Convert risk rating format from HIGH_RISK to HIGH RISK
                        if parsed_json["risk_rating"] in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
                            parsed_json["risk_rating"] = parsed_json["risk_rating"].replace("_", " ")
                        else:
                            parsed_json["risk_rating"] = "NO RISK RATING PROVIDED"
                        return parsed_json

                except json.JSONDecodeError:
                    print("Failed to parse JSON response")

            return self._create_fallback_response("Failed to get valid analysis")

        except Exception as e:
            print(f"Error in LLM query: {str(e)}")
            return self._create_fallback_response(str(e))

    def _create_fallback_response(self, error_msg: str) -> Dict:
        return {
            "detailed_risk_analysis": f"Analysis failed: {error_msg}",
            "major_risk_factors": ["Analysis incomplete"],
            "mitigating_factors": ["Technical error occurred"],
            "risk_rating": "NO RISK RATING PROVIDED"
        }

    def format_record(self, record: Dict) -> str:
        # Normalize Aadhaar address fields to Yes/No
        if 'Aadhaar Present Address' in record:
            record['Aadhaar Present Address'] = 'Yes' if str(record['Aadhaar Present Address']).lower() == 'yes' else 'No'
        if 'Aadhaar Permanent Address' in record:
            record['Aadhaar Permanent Address'] = 'Yes' if str(record['Aadhaar Permanent Address']).lower() == 'yes' else 'No'
            
        important_fields = [
            "Credit Score",
            "DTI",
            "Debt",
            "Loan Repayment History",
            "Employement Type",
            "Annual Income",
            "Net Monthly Income",
            "Loan Type",
            "Loan Purpose",
            "Loan Amount",
            "Constitution",
            "Dependents",
            "Aadhaar Present Address",
            "Aadhaar Permanent Address",
            "Insurance Interest",
            "Reference 1 Relationship",
            "Litigation History"
        ]

        return "\n".join(f"{field}: {record[field]}" for field in important_fields if field in record)

    def analyze_record(self, record: Dict) -> Dict:
        record_str = self.format_record(record)
        prompt = self.analysis_prompt.format(record=record_str)
        return self.query_llm(prompt)

    def process_records(self, input_file: str) -> str:
        try:
            print_step_header("Risk Analysis")

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

                print_risk_summary(
                    analysis.get('risk_rating', 'UNKNOWN'),
                    analysis.get('major_risk_factors', []),
                    analysis.get('mitigating_factors', [])
                )

                time.sleep(1)

            output_file = "risk_analysis_results.csv"
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