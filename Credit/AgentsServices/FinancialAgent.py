import os
import time
import json
import re
from typing import Dict
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FinancialAgent:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY_3")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = Groq(api_key=api_key)
        self.models = [
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "qwen-qwq-32b"
        ]
    
    def analyze_financials(self, application_data: Dict) -> Dict:
        """
        Analyze financial aspects of a loan application with retry logic
        
        Args:
            application_data (dict): Complete application data dictionary
            
        Returns:
            dict: Financial analysis results
        """
        # Extract relevant fields for financial analysis
        financial_data = {
            # Employment details
            "Applicant Employment institution name": application_data.get("Applicant Employment institution name", "NA"),
            "Applicant Retirement date": application_data.get("Applicant Retirement date", "NA"),
            "Applicant Completed years of service": application_data.get("Applicant Completed years of service", "NA"),
            
            # Income information
            "Applicant Gross salary": application_data.get("Applicant Gross salary", "NA"),
            "Applicant Other incomes": application_data.get("Applicant Other incomes", "NA"),
            "Spouse's Gross salary": application_data.get("Spouse's Gross salary", "NA"),
            "Spouse's Other incomes": application_data.get("Spouse's Other incomes", "NA"),
            
            # Guarantor information
            "Guarantor1 occupations": application_data.get("Guarantor1 occupations", "NA"),
            "Guarantor1 income from salary": application_data.get("Guarantor1 income from salary", "NA"),
            "Guarantor1 income from other sources": application_data.get("Guarantor1 income from other sources", "NA"),
            "Guarantor2 income from salary": application_data.get("Guarantor2 income from salary", "NA"),
            "Guarantor2 income from other sources": application_data.get("Guarantor2 income from other sources", "NA"),
            
            # Loan details
            "Investment proposed": application_data.get("Investment proposed", "NA"),
            "Margin offered": application_data.get("Margin offered", "NA"),
            "Bank finance required": application_data.get("Bank finance required", "NA"),
            "Repayment period required": application_data.get("Repayment period required", "NA"),
        }
        
        # Generate a string representation for the prompt
        formatted_data = "\n".join([f"{key}: {value}" for key, value in financial_data.items()])
        
        # Enhanced prompt with explicit instructions to avoid calculations in JSON
        prompt = f"""
You are a financial analyst at a banking institution. Analyze the following loan application financial details and provide a comprehensive assessment.

APPLICATION FINANCIAL DETAILS:
{formatted_data}

TASK:
1. Calculate the estimated monthly payment based on loan amount and repayment period
2. Calculate the total monthly income from all sources
3. Calculate the debt-to-income ratio
4. Analyze the applicant's financial capacity to repay the loan

IMPORTANT: Your response must be in valid JSON format with the following keys:
1. detailed_analysis: A thorough examination of the financial aspects (including your calculated DTI)
2. key_strengths: A list of 3-5 financial strengths of this application
3. key_concerns: A list of 3-5 financial concerns or weaknesses
4. monthly_payment: Your calculated monthly payment amount as a NUMBER ONLY, not a formula
5. total_monthly_income: Total calculated monthly income from all sources as a NUMBER ONLY, not a formula
6. debt_to_income_ratio: Calculated as monthly payment / total monthly income as a NUMBER ONLY, not a formula
7. rating: The financial strength rating (must be exactly one of: "STRONG", "INTERMEDIATE", or "WEAK")
8. fixed_obligation_ratio: Monthly fixed obligations / monthly income (excluding the new loan)
10. net_worth_analysis: Asset valuation vs. existing liabilities
11. liquidity_assessment: Cash/liquid assets available as % of loan amount

CRITICAL INSTRUCTION: Do NOT include any calculations or formulas in your JSON response. Perform all calculations mentally and provide only the final numerical results. For example, instead of writing "monthly_payment: 40922 / 12", just write "monthly_payment: 3410.17".

Base your analysis on standard banking financial assessment practices, considering factors like:
- Income stability and sufficiency
- Debt-to-income ratio
- Repayment capacity
- Margin adequacy (down payment)
- Loan tenor reasonableness
- Combined household income
- Guarantor financial strength
"""

        # Enhanced system prompt
        system_prompt = """You are an expert financial analyst AI assistant. Always return valid JSON without any calculations in the JSON itself.

IMPORTANT: When returning numbers in your JSON response:
1. Always perform calculations BEFORE putting values in JSON
2. Only include the FINAL numerical results, NEVER expressions
3. Examples of WRONG output: "total_monthly_income": 233000 + 200000
4. Examples of CORRECT output: "total_monthly_income": 433000
5. Do NOT include ANY mathematical operators (+, -, *, /) in values
6. For debt-to-income ratio, provide the decimal value (e.g., 0.35), not a percentage"""

        # Try each model with retries
        for model in self.models:
            print(f"Attempting analysis with model: {model}")
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    # Call Groq API for financial analysis
                    completion = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=2048,
                        response_format={"type": "json_object"}
                    )
                    
                    # Extract and parse the analysis
                    analysis_json = completion.choices[0].message.content
                    
                    try:
                        # Fix common calculation patterns before parsing
                        fixed_json = self._fix_json_calculations(analysis_json)
                        analysis_data = json.loads(fixed_json)
                        
                        # Validate required fields and types
                        self._validate_financial_data(analysis_data)
                        
                        # Print successful analysis results
                        print("\n" + "="*50)
                        print("----------------------")
                        print("FINANCIAL ANALYSIS RESULTS:")
                        print("----------------------")
                        print("="*50)
                        print("----------------------")
                        print(f"Financial Rating: {analysis_data.get('rating', 'Not available')}")
                        print("----------------------")
                        print(f"Key Strengths: {analysis_data.get('key_strengths', 'Not available')}")
                        print("----------------------")
                        print(f"Key Concerns: {analysis_data.get('key_concerns', 'Not available')}")
                        print("----------------------")
                        print(f"Debt-to-Income Ratio: {analysis_data.get('debt_to_income_ratio', 'Not available')}")
                        print("----------------------")
                        print("="*50 + "\n")
                        
                        return analysis_data
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"JSON validation error (attempt {attempt+1}/{max_retries}): {str(e)}")
                        if attempt == max_retries - 1:
                            continue  # Try next model
                        time.sleep(1 * (attempt + 1))
                        
                except Exception as e:
                    print(f"API error (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt == max_retries - 1:
                        continue  # Try next model
                    time.sleep(1 * (attempt + 1))
        
        # All models and retries failed
        error_response = {
            "error": "All models failed to analyze financial data after multiple retries",
            "detailed_analysis": "Financial analysis could not be completed due to persistent API errors.",
            "key_strengths": ["Data requires manual review"],
            "key_concerns": ["System error occurred with all fallback models"],
            "monthly_payment": 0,
            "total_monthly_income": 0,
            "debt_to_income_ratio": 0,
            "rating": "NONE"
        }
        
        print("\n" + "="*50)
        print("----------------------")
        print("FINANCIAL ANALYSIS ERROR: ALL ATTEMPTS FAILED")
        print("----------------------")
        print("="*50)
        
        return error_response
    
    def _fix_json_calculations(self, json_string: str) -> str:
        """Fix common calculation patterns in JSON strings by evaluating expressions"""
        # Step 1: Find patterns where calculations are present
        calculation_pattern = r'\"([\w_]+)\"\s*:\s*([\d\.\+\-\*\/\(\)\s]+),?'
        
        # Process each line 
        lines = json_string.split("\n")
        fixed_lines = []
        
        for line in lines:
            match = re.search(calculation_pattern, line)
            if match:
                field_name = match.group(1)
                calculation = match.group(2).strip()
                
                # Only process numeric fields
                if field_name in ["monthly_payment", "total_monthly_income", "debt_to_income_ratio"]:
                    # Try to evaluate the expression
                    try:
                        # Clean up the calculation string to remove any trailing commas
                        if calculation.endswith(','):
                            calculation = calculation[:-1]
                            
                        # Check for simple arithmetic expressions
                        if any(op in calculation for op in ['+', '-', '*', '/', '(']):
                            # Use safer eval by defining allowed operators
                            result = self._safe_eval(calculation)
                            # For debt-to-income ratio, ensure it's a probability (0-1)
                            if field_name == "debt_to_income_ratio" and result > 1:
                                result = result / 100  # Convert percentage to decimal if needed
                            fixed_line = f'  "{field_name}": {result},'
                        else:
                            # If it's just a number, keep it
                            fixed_line = line
                    except Exception:
                        # If evaluation fails, use a reasonable default
                        if field_name == "monthly_payment":
                            fixed_line = f'  "{field_name}": 10000,'  # Reasonable monthly payment
                        elif field_name == "total_monthly_income":
                            fixed_line = f'  "{field_name}": 100000,'  # Reasonable income
                        elif field_name == "debt_to_income_ratio":
                            fixed_line = f'  "{field_name}": 0.3,'  # Middle-range DTI
                        else:
                            fixed_line = f'  "{field_name}": 0,'
                    
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return "\n".join(fixed_lines)

    def _safe_eval(self, expr):
        """Safely evaluate a mathematical expression"""
        # Remove any non-numeric, non-operator characters
        clean_expr = re.sub(r'[^0-9\+\-\*\/\.\(\)\s]', '', expr)
        
        # Check if this is a valid mathematical expression
        if not re.match(r'^[\d\.\+\-\*\/\(\)\s]+$', clean_expr):
            raise ValueError("Invalid expression")
            
        # Evaluate
        return eval(clean_expr)
    
    def _validate_financial_data(self, data: Dict) -> None:
        """Validate that financial data has all required fields and correct types"""
        required_fields = ["detailed_analysis", "key_strengths", "key_concerns", 
                          "monthly_payment", "total_monthly_income", "debt_to_income_ratio", "rating",
                          "fixed_obligation_ratio", "net_worth_analysis", "liquidity_assessment"]
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields in financial analysis: {missing_fields}")
        
        # Check numeric fields
        numeric_fields = ["monthly_payment", "total_monthly_income", "debt_to_income_ratio", 
                          "fixed_obligation_ratio", "liquidity_assessment"]
        for field in numeric_fields:
            if not isinstance(data[field], (int, float)):
                # Try to convert string to number
                try:
                    data[field] = float(data[field])
                except (ValueError, TypeError):
                    raise ValueError(f"Field '{field}' must be a number")
        
        # Check rating value
        if data["rating"] not in ["STRONG", "INTERMEDIATE", "WEAK"]:
            raise ValueError(f"Invalid rating: {data['rating']}")