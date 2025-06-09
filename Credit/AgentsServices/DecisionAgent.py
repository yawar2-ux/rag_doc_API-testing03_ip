import os
import json
from typing import Dict
from collections import OrderedDict
from groq import Groq
from dotenv import load_dotenv
from .RiskAgent import RiskAgent
from .FinancialAgent import FinancialAgent

"""
risk and financial analysis agent

output:
    Risk_Rating - From risk assessment
    Financial_Rating - From financial assessment
    final_analysis - Detailed combined analysis
    decision - Either "APPROVED" or "REJECTED"
    confidence_level - "HIGH", "MEDIUM", or "LOW"
    key_decision_factors - List of factors that led to the decision
    key_concern_factors - List of specific concerns about the application
"""

# Load environment variables
load_dotenv()

class DecisionAgent:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY_3")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.client = Groq(api_key=api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"  # You can use a more powerful model if needed
        self.risk_agent = RiskAgent()
        self.financial_agent = FinancialAgent()
    
    def print_decision(self, decision_data):
        """Helper method to print decision results consistently"""
        print("\n" + "="*50)
        print("----------------------")
        print("FINAL DECISION RESULTS:")
        print("----------------------")
        print("="*50)
        print("----------------------")
        print(f"Financial Rating: {decision_data.get('Financial_Rating', 'Not available')}")
        print("----------------------")
        print(f"Risk Rating: {decision_data.get('Risk_Rating', 'Not available')}")
        print("----------------------")
        print(f"Decision: {decision_data.get('decision', 'Not available')}")
        print("----------------------")
        print(f"Confidence Level: {decision_data.get('confidence_level', 'Not available')}")
        print("----------------------")
        print(f"Final Analysis: {decision_data.get('final_analysis', 'Not available')}")
        print("----------------------")
        print(f"Key Decision Factors: {decision_data.get('key_decision_factors', 'Not available')}")
        print("----------------------")
        print(f"Key Concern Factors: {decision_data.get('key_concern_factors', 'Not available')}")
        print("----------------------")
        print("="*50 + "\n")
    
    def make_decision(self, application_data: Dict) -> Dict:
        """
        Make final loan decision by considering both risk and financial analyses
        
        Args:
            application_data: Original application data dictionary
            
        Returns:
            Dict: Final decision with analysis
        """
        # Get risk analysis
        risk_analysis = self.risk_agent.analyze_risk(application_data)
        if isinstance(risk_analysis, str):
            risk_analysis = json.loads(risk_analysis)
                
        # Get financial analysis
        financial_analysis = self.financial_agent.analyze_financials(application_data)
        if isinstance(financial_analysis, str):
            financial_analysis = json.loads(financial_analysis)
        
        # Extract ratings for context
        risk_rating = risk_analysis.get("risk_rating", "NONE")
        financial_rating = financial_analysis.get("rating", "NONE")
        
        # Check if we can proceed with analysis
        if "error" in risk_analysis or "error" in financial_analysis or risk_rating == "NONE" or financial_rating == "NONE":
            # Handle error case but don't make a decision
            return {
                "Financial_Rating": financial_rating,
                "Risk_Rating": risk_rating,
                "decision": "NONE",
                "confidence_level": "NONE",
                "final_analysis": "Decision could not be completed due to missing analysis data.",
                "key_decision_factors": ["Incomplete analysis data"],
                "key_concern_factors": ["Analysis services failed to provide complete assessment"]
            }
        
        # Basic application info for context
        basic_info = {
            "Purpose of loan": application_data.get("Purpose of loan", "NA"),
            "Applicant Name": application_data.get("Applicant Name", "NA"),
            "Bank finance required": application_data.get("Bank finance required", "NA"),
            "Repayment period required": application_data.get("Repayment period required", "NA"),
            "Collateral land/building details": application_data.get("Collateral land/building details", "NA"),
        }
        
        # Format data for prompt
        formatted_basic_info = "\n".join([f"{key}: {value}" for key, value in basic_info.items()])
        
        # Create prompt that provides guidance but allows the LLM to make the actual decision
        prompt = f"""
You are the head of a credit committee at a banking institution. Review the risk and financial analyses for this loan application and make a final credit decision.

BASIC APPLICATION INFO:
{formatted_basic_info}

RISK ANALYSIS:
Risk Rating: {risk_rating}

Detailed Risk Analysis:
{risk_analysis.get('detailed_risk_analysis', 'Not provided')}

Major Risk Factors:
{', '.join(risk_analysis.get('major_risk_factors', ['Not provided']))}

Mitigating Factors:
{', '.join(risk_analysis.get('mitigating_factors', ['Not provided']))}

FINANCIAL ANALYSIS:
Financial Rating: {financial_rating}

Detailed Financial Analysis:
{financial_analysis.get('detailed_analysis', 'Not provided')}

Financial Metrics:
- Monthly Payment: {financial_analysis.get('monthly_payment', 'Not calculated')}
- Total Monthly Income: {financial_analysis.get('total_monthly_income', 'Not calculated')}
- Debt-to-Income Ratio: {financial_analysis.get('debt_to_income_ratio', 'Not calculated')}

Key Financial Strengths:
{', '.join(financial_analysis.get('key_strengths', ['Not provided']))}

Key Financial Concerns:
{', '.join(financial_analysis.get('key_concerns', ['Not provided']))}

BANKING GUIDANCE:
- Risk Rating (HIGH RISK, MEDIUM RISK, LOW RISK) represents the likelihood of default
- Financial Rating (STRONG, INTERMEDIATE, WEAK) represents repayment capacity
- Typical bank policies suggest caution with HIGH RISK applications or WEAK financials
- A debt-to-income ratio above 50% is generally concerning
- Fixed obligation ratio above 70% typically requires rejection

YOUR TASK:
As credit committee head, you have full authority to make the final decision. Consider both risk and financial factors balanced with banking best practices. You are not bound by rigid rules - use your best judgment.

Provide your final decision in JSON format with the following keys:
1. Financial_Rating: The financial rating from the financial assessment (preserve as is)
2. Risk_Rating: The risk rating from the risk assessment (preserve as is)
3. decision: Your final decision (must be exactly "APPROVED" or "REJECTED")
4. confidence_level: How confident you are in this decision (must be exactly "HIGH", "MEDIUM", or "LOW")
5. final_analysis: A detailed explanation of your decision-making process
6. key_decision_factors: List of 3-5 most important factors that led to your decision
7. key_concern_factors: List of 3-5 specific concerns about this application, regardless of approval status
"""

        try:
            # Call Groq API for decision making - let the LLM decide
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior credit committee head at a bank making final loan approval decisions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            
            # Extract the decision - trust the LLM's judgment
            decision_json = completion.choices[0].message.content
            
            # Parse JSON to dictionary before returning
            decision_data = json.loads(decision_json) if isinstance(decision_json, str) else decision_json
            
            # Ensure correct order with OrderedDict
            ordered_decision = OrderedDict([
                ("Financial_Rating", decision_data.get("Financial_Rating")),
                ("Risk_Rating", decision_data.get("Risk_Rating")), 
                ("decision", decision_data.get("decision")),
                ("confidence_level", decision_data.get("confidence_level")), 
                ("final_analysis", decision_data.get("final_analysis")),
                ("key_decision_factors", decision_data.get("key_decision_factors")),
                ("key_concern_factors", decision_data.get("key_concern_factors"))
            ])
            
            # Print decision results
            self.print_decision(ordered_decision)
            
            return ordered_decision
            
        except Exception as e:
            # Only handle errors, not actual decisions
            error_response = OrderedDict([
                ("Financial_Rating", financial_rating),
                ("Risk_Rating", risk_rating),
                ("decision", "NONE"),
                ("confidence_level", "NONE"), 
                ("final_analysis", "Decision could not be completed due to a system error."),
                ("key_decision_factors", ["System error occurred"]),
                ("key_concern_factors", ["System error prevented proper analysis"]),
                ("error", f"Error making credit decision: {str(e)}")
            ])
            
            self.print_decision(error_response)
            
            return error_response