import pandas as pd
import json
import requests
import time
import os
from typing import Dict
from groq import Groq
from credit_underwriting.utils import print_step_header, print_record_status, print_decision_summary, print_completion_summary

class DecisionAgent:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.analysis_prompt = """You are an expert credit underwriting agent. Review both the financial analysis and risk assessment for this loan application and make a final decision.

FINANCIAL ANALYSIS:
{financial_data}

RISK ASSESSMENT:
{risk_data}

As a credit underwriting agent, analyze:

1. Overall Financial Health
   - Financial rating: {financial_rating}
   - Key financial strengths and concerns
   - Income stability and debt management
   - Savings and assets

2. Risk Profile
   - Risk rating: {risk_rating}
   - Major risk factors identified
   - Mitigating factors present
   - Credit history and behavior

3. Loan Specifics
   - Loan amount feasibility
   - Purpose alignment
   - Repayment capacity
   - Collateral/Security (if applicable)

4. Decision Criteria
   - Debt-to-income ratio
   - Credit score evaluation
   - Employment stability
   - Asset quality
   - Past repayment behavior
   - Reference checks

Analyze all aspects thoroughly and provide your decision in this exact JSON format:
{{
    "final_analysis": "your detailed combined analysis here",
    "decision": "APPROVED/REJECTED",
    "confidence_level": "HIGH/MEDIUM/LOW",
    "key_decision_factors": ["factor1", "factor2"],
    "conditions_if_approved": ["condition1", "condition2"],
    "rejection_reasons_if_rejected": ["reason1", "reason2"]
}}"""

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
            
            print("\n\nDecision Agent ==> ", response_text)

            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                try:
                    json_str = response_text[start_idx:end_idx]
                    parsed_json = json.loads(json_str)

                    required_fields = [
                        "final_analysis",
                        "decision",
                        "confidence_level",
                        "key_decision_factors"
                    ]

                    if all(field in parsed_json for field in required_fields):
                        if parsed_json["decision"] not in ["APPROVED", "REJECTED"]:
                            parsed_json["decision"] = "NO DECISION MADE"
                        if parsed_json["confidence_level"] not in ["HIGH", "MEDIUM", "LOW"]:
                            parsed_json["confidence_level"] = "NO CONFIDENCE LEVEL"
                        return parsed_json

                except json.JSONDecodeError:
                    print("Failed to parse JSON response")

            return self._create_fallback_response("Failed to get valid analysis")

        except Exception as e:
            print(f"Error in LLM query: {str(e)}")
            return self._create_fallback_response(str(e))

    def _create_fallback_response(self, error_msg: str) -> Dict:
        return {
            "final_analysis": f"Analysis failed: {error_msg}",
            "decision": "REJECTED",
            "confidence_level": "LOW",
            "key_decision_factors": ["Analysis failed"],
            "conditions_if_approved": [],
            "rejection_reasons_if_rejected": ["System unable to make automated decision"]
        }

    def make_decision(self, financial_data: Dict, risk_data: Dict) -> Dict:
        prompt = self.analysis_prompt.format(
            financial_data=json.dumps(financial_data, indent=2),
            risk_data=json.dumps(risk_data, indent=2),
            financial_rating=financial_data.get('rating', 'UNKNOWN'),
            risk_rating=risk_data.get('risk_rating', 'UNKNOWN')
        )
        return self.query_llm(prompt)

    def process_applications(self, financial_file: str, risk_file: str) -> str:
        try:
            print_step_header("Final Decision Making")

            financial_df = pd.read_csv(financial_file)
            risk_df = pd.read_csv(risk_file)

            # Normalize Aadhaar address fields
            for df in [financial_df, risk_df]:
                if 'Aadhaar Present Address' in df.columns:
                    df['Aadhaar Present Address'] = df['Aadhaar Present Address'].apply(lambda x: 'Yes' if str(x).lower() == 'yes' else 'No')
                if 'Aadhaar Permanent Address' in df.columns:
                    df['Aadhaar Permanent Address'] = df['Aadhaar Permanent Address'].apply(lambda x: 'Yes' if str(x).lower() == 'yes' else 'No')

            financial_df['Customer_ID'] = financial_df['Customer_ID'].astype(str)
            risk_df['Customer_ID'] = risk_df['Customer_ID'].astype(str)

            results = []

            for idx, cust_id in enumerate(financial_df['Customer_ID'].unique(), 1):
                print_record_status(idx, len(financial_df), f"Customer {cust_id}")

                financial_record = financial_df[financial_df['Customer_ID'] == cust_id].iloc[0].to_dict()
                risk_record = risk_df[risk_df['Customer_ID'] == cust_id].iloc[0].to_dict()

                decision = self.make_decision(financial_record, risk_record)
                result = {
                    "Customer_ID": cust_id,
                    "Financial_Rating": financial_record.get('rating', 'UNKNOWN'),
                    "Risk_Rating": risk_record.get('risk_rating', 'UNKNOWN'),
                    **decision
                }

                results.append(result)
                print_decision_summary(decision)

                time.sleep(1)

            output_file = "final_loan_decisions.csv"
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