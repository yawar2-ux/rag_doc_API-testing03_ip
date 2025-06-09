import os
from typing import Dict
from groq import Groq
from dotenv import load_dotenv
import json

"""
input:
    Purpose of loan
    Applicant Existing Loan details
    Applicant Landed property details
    Spouse's Existing Loan details
    Spouse's Landed property details
    Guarantor1 loan details
    Guarantor1 landed assets details
    Guarantor2 loan details
    Guarantor2 landed assets details
    Collateral land/building details
    LIC policy details
    NSC/KVP/Bank/Post Office deposit details

output:
    validation_issues - List of validation problems found
    detailed_risk_analysis - Thorough analysis of all risk factors
    major_risk_factors - List of identified risk factors
    mitigating_factors - List of factors mitigating the risks
    risk_rating - The overall risk rating (HIGH RISK, MEDIUM RISK, or LOW RISK)
"""

# Load environment variables
load_dotenv()

class RiskAgent:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY_3")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = Groq(api_key=api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    def analyze_risk(self, application_data: Dict) -> Dict:
        """
        Analyze risk factors in a loan application
        
        Args:
            application_data (dict): Complete application data dictionary
            
        Returns:
            dict: Risk analysis results
        """
        # Extract relevant fields for risk analysis
        risk_data = {
            # Basic applicant information
            "Applicant Name": application_data.get("Applicant Name", "NA"),
            "Applicant Address": application_data.get("Applicant Address", "NA"),
            "Applicant Age": application_data.get("Applicant Age", "NA"),
            "Applicant Employment institution name": application_data.get("Applicant Employment institution name", "NA"),
            "Applicant Retirement date": application_data.get("Applicant Retirement date", "NA"),
            
            # Existing fields
            "Purpose of loan": application_data.get("Purpose of loan", "NA"),
            "Applicant Existing Loan details": application_data.get("Applicant Existing Loan details", "NA"),
            "Applicant Landed property details": application_data.get("Applicant Landed property details", "NA"),
            "Spouse's Existing Loan details": application_data.get("Spouse's Existing Loan details", "NA"),
            "Spouse's Landed property details": application_data.get("Spouse's Landed property details", "NA"),
            
            # Financial information
            "Applicant Gross salary": application_data.get("Applicant Gross salary", "NA"),
            "Applicant Other incomes": application_data.get("Applicant Other incomes", "NA"),
            "Spouse's Gross salary": application_data.get("Spouse's Gross salary", "NA"),
            "Spouse's Other incomes": application_data.get("Spouse's Other incomes", "NA"),
            
            # Guarantor information
            "Guarantor1 name": application_data.get("Guarantor1 name", "NA"),
            "Guarantor1 relationship with applicant": application_data.get("Guarantor1 relationship with applicant", "NA"),
            "Guarantor1 loan details": application_data.get("Guarantor1 loan details", "NA"),
            "Guarantor1 landed assets details": application_data.get("Guarantor1 landed assets details", "NA"),
            "Guarantor2 loan details": application_data.get("Guarantor2 loan details", "NA"),
            "Guarantor2 landed assets details": application_data.get("Guarantor2 landed assets details", "NA"),
            
            # Security and collateral
            "Collateral land/building details": application_data.get("Collateral land/building details", "NA"),
            "LIC policy details": application_data.get("LIC policy details", "NA"),
            "NSC/KVP/Bank/Post Office deposit details": application_data.get("NSC/KVP/Bank/Post Office deposit details", "NA"),
            
            # Loan details
            "Bank finance required": application_data.get("Bank finance required", "NA"),
            "Repayment period required": application_data.get("Repayment period required", "NA"),
        }
        
        # Format data for prompt
        formatted_data = "\n".join([f"{key}: {value}" for key, value in risk_data.items()])
        
        # Create enhanced prompt for risk analysis with validation instructions
        prompt = f"""
You are a senior risk analyst at a banking institution. Analyze the following loan application details and provide a comprehensive risk assessment.

APPLICATION DETAILS:
{formatted_data}

IMPORTANT VALIDATION CHECKS (Perform these first):
1. AGE VALIDATION:
   - Check if applicant age is below 18 (underage)
   - Check if applicant age is above 75 (overage risk)
   
2. RETIREMENT DATE VALIDATION:
   - Check if retirement date is in the past
   - Compare retirement date with the repayment period to verify the loan won't extend beyond retirement
   
3. LOAN-TO-INCOME VALIDATION:
   - Calculate the ratio of loan amount to monthly income
   - Flag if loan amount exceeds 5 years of total monthly income (60 months)

4. LOAN-TO-COLLATERAL VALIDATION:
   - Calculate ratio of loan amount to collateral value
   - Flag if loan exceeds 70-80% of collateral value

5. INDUSTRY/PURPOSE VALIDATION:
   - Check if loan purpose falls into high-risk category
   - Flag loans for industries on bank's caution list

First identify any validation issues, then incorporate them into your overall risk assessment.

Provide your analysis in JSON format with the following keys:
1. validation_issues: List any validation problems found (empty list if none found)
2. detailed_risk_analysis: A thorough paragraph-by-paragraph analysis of all risk factors
3. major_risk_factors: A list of 3-5 identified risk factors, in order of severity
4. mitigating_factors: A list of 3-5 factors that mitigate the risks
5. risk_rating: The overall risk rating (must be exactly one of: "HIGH RISK", "MEDIUM RISK", or "LOW RISK")

If serious validation issues are found (like underage applicant), these should be reflected in your risk rating and listed among major risk factors.

Base your analysis on standard banking risk assessment practices.
"""

        # Enhanced system prompt
        system_prompt = """You are an expert banking risk analyst. 
        
VALIDATION RULES TO STRICTLY APPLY:
- Applicants must be at least 18 years old
- Applicants should generally be under 75 years old
- Retirement date must be in the future
- Loan repayment period should not extend beyond retirement date
- Loan amount should be reasonable compared to income

Always check these validation rules first before completing your risk analysis."""
        
        try:
            # Call Groq API for risk analysis
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            # Extract the analysis
            analysis_json = completion.choices[0].message.content
            
            # Parse the JSON string to a dictionary
            risk_data = json.loads(analysis_json) if isinstance(analysis_json, str) else analysis_json
            
            # Print risk analysis results including validation issues
            print("\n" + "="*50)
            print("----------------------")
            print("RISK ANALYSIS RESULTS:")
            print("----------------------")
            print("="*50)
            print("----------------------")
            if "validation_issues" in risk_data and risk_data["validation_issues"]:
                print("VALIDATION ISSUES DETECTED:")
                for issue in risk_data["validation_issues"]:
                    print(f"- {issue}")
                print("----------------------")
            print(f"Risk Rating: {risk_data.get('risk_rating', 'Not available')}")
            print("----------------------")
            print(f"Major Risk Factors: {risk_data.get('major_risk_factors', 'Not available')}")
            print("----------------------")
            print(f"Detailed Risk Analysis: {risk_data.get('detailed_risk_analysis', 'Not available')}")
            print("----------------------")
            print(f"Mitigating Factors: {risk_data.get('mitigating_factors', 'Not available')}")
            print("----------------------")
            print("="*50 + "\n")
            
            return risk_data
            
        except Exception as e:
            error_response = {
                "error": f"Error performing risk analysis: {str(e)}",
                "validation_issues": ["System error prevented validation"],
                "detailed_risk_analysis": "Analysis could not be completed due to an error.",
                "major_risk_factors": ["System error occurred"],
                "mitigating_factors": [],
                "risk_rating": "HIGH RISK"
            }
            
            # Print error with all required fields
            print("\n" + "="*50)
            print("----------------------")
            print("RISK ANALYSIS ERROR:")
            print("----------------------")
            print("="*50)
            print("----------------------")
            print(f"Error: {str(e)}")
            print("----------------------")
            print("="*50 + "\n")
            
            return error_response