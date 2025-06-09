import pandas as pd
import os
import json
from collections import OrderedDict
from ..Services import FieldConfig
from . import DocAgent
from .RiskAgent import RiskAgent
from .FinancialAgent import FinancialAgent  
from .DecisionAgent import DecisionAgent

def process_applications(csv_file_path, analyze_complete=True):
    """
    Process all applications in a CSV file, validate data, and run complete applications 
    through risk, financial, and decision agents.
    
    Args:
        csv_file_path (str): Path to the CSV file containing application data
        analyze_complete (bool): Whether to analyze complete applications with additional agents
        
    Returns:
        dict: Results with incomplete and complete applications (with analysis if applicable)
    """
    # Validate the CSV file exists
    if not os.path.exists(csv_file_path):
        return {"status": "error", "message": f"File not found: {csv_file_path}"}
    
    try:
        # Read the CSV file, explicitly treating "None" as strings by using na_filter=False
        df = pd.read_csv(csv_file_path, skip_blank_lines=True, na_filter=False)
        
        if "Application" not in df.columns:
            return {"status": "error", "message": "CSV file missing 'Application' column"}
        
        # Process each application using DocAgent
        results = []
        
        for _, row in df.iterrows():
            # Convert row to dictionary with proper JSON serialization
            app_data = {}
            for col in row.index:
                value = row[col]
                
                # Handle empty strings
                if isinstance(value, str) and value.strip() == "":
                    app_data[col] = "NA"
                # Handle explicit "NA" strings
                elif isinstance(value, str) and value == "NA":
                    app_data[col] = "NA"
                # Keep "None" strings as is - they should be considered valid/complete
                elif isinstance(value, str) and value == "None":
                    app_data[col] = "None"
                # Convert complex types to strings for serialization
                else:
                    app_data[col] = str(value) if not isinstance(value, (int, float, bool, str)) else value
            
            # Process application using DocAgent
            application_result = DocAgent.process_application(app_data, FieldConfig.keys)
            results.append(application_result)
        
        # Categorize results into incomplete and complete applications
        incomplete_applications = []
        complete_applications = []
        
        for r in results:
            if r["has_missing_fields"] or r["has_invalid_types"] or r["has_invalid_financial_structure"]:
                # Determine status message based on validation issues
                status_parts = []
                if r["has_missing_fields"]:
                    status_parts.append("Incomplete")
                if r["has_invalid_types"]:
                    status_parts.append("Invalid Fields")
                if r["has_invalid_financial_structure"]:
                    status_parts.append("Invalid Financial Structure")
                
                # Join status parts with "&"
                r["application_data"]["status"] = " & ".join(status_parts)
                incomplete_applications.append(r)
            else:
                # No issues, complete application
                r["application_data"]["status"] = "Complete"
                complete_applications.append(r)
        
        # Create simplified format for incomplete applications
        simplified_incomplete = []
        for r in incomplete_applications:
            # Use OrderedDict to ensure field order
            simplified_item = OrderedDict()
            
            # Add fields in the desired order
            simplified_item["Application"] = r["Application"]
            simplified_item["status"] = r["application_data"]["status"]
            
            # Add missing fields if present
            if r["has_missing_fields"]:
                simplified_item["missing_fields"] = r["missing_fields"]
                
            # Add invalid fields if present
            if r["has_invalid_types"]:
                simplified_item["invalid_type_fields"] = r["invalid_type_fields"]
            
            # Add invalid financial structure details if present
            if r["has_invalid_financial_structure"]:
                simplified_item["invalid_financial_structure"] = r["invalid_financial_structure"]
                
            simplified_incomplete.append(simplified_item)
        
        # Process complete applications with analysis agents if requested
        analyzed_complete_apps = []
        
        if analyze_complete and complete_applications:
            # Initialize the decision agent (which internally uses risk and financial agents)
            decision_agent = DecisionAgent()
            
            for r in complete_applications:
                app_data = r["application_data"]
                app_id = r["Application"]
                
                try:
                    # Get the final decision (this calls both risk and financial agents internally)
                    decision_result = decision_agent.make_decision(app_data)
                    
                    # Add decision analysis to application data
                    app_data["decision_analysis"] = decision_result
                    app_data["processing_status"] = "Analysis Complete"
                except Exception as e:
                    # Handle errors in analysis
                    app_data["processing_status"] = f"Analysis Error: {str(e)}"
                
                # Reorder fields to put Application at the top
                if "Application" in app_data:
                    ordered_app_data = OrderedDict()
                    # Add Application field first
                    ordered_app_data["Application"] = app_data["Application"]
                    # Add all other fields in their original order
                    for key, value in app_data.items():
                        if key != "Application":
                            ordered_app_data[key] = value
                    
                    analyzed_complete_apps.append(ordered_app_data)
                else:
                    analyzed_complete_apps.append(app_data)
        else:
            # If analysis not requested, just return the application data with reordered fields
            for r in complete_applications:
                app_data = r["application_data"]
                if "Application" in app_data:
                    ordered_app_data = OrderedDict()
                    # Add Application field first
                    ordered_app_data["Application"] = app_data["Application"]
                    # Add all other fields in their original order
                    for key, value in app_data.items():
                        if key != "Application":
                            ordered_app_data[key] = value
                    
                    analyzed_complete_apps.append(ordered_app_data)
                else:
                    analyzed_complete_apps.append(app_data)
        
        # Return simplified format
        return {
            "status": "completed",
            "csv_file": csv_file_path,
            "processing_results": {
                "incomplete": simplified_incomplete,
                "complete": analyzed_complete_apps
            }
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error processing applications: {str(e)}"}

def analyze_single_application(application_data):
    """
    Analyze a single complete application with all agents.
    
    Args:
        application_data (dict): Complete application data
        
    Returns:
        dict: Application data with decision analysis results
    """
    try:
        # First, check financial structure validity
        app_id = application_data.get("Application", "UNKNOWN")
        field_keys = list(application_data.keys())
        
        doc_result = DocAgent.process_application(application_data, field_keys)
        
        # Don't proceed with analysis if financial structure is invalid
        if doc_result["has_invalid_financial_structure"]:
            print(f"Application {app_id} has invalid financial structure. Skipping analysis.")
            return {
                "Application": app_id,
                "status": "Incomplete",
                "decision": "REJECTED",
                "reason": "Invalid financial structure",
                "invalid_financial_structure": doc_result["invalid_financial_structure"]
            }
            
        # Continue with risk and financial analysis if financial structure is valid
        risk_agent = RiskAgent()
        financial_agent = FinancialAgent()
        decision_agent = DecisionAgent()
        
        # Proceed with analysis...
        
    except Exception as e:
        # Exception handling...
        print("\n" + "="*50)
        print("----------------------")
        print("ORCHESTRATION ERROR - DECISION AGENT DID NOT HANDLE:")
        print("----------------------")
        print(f"Error: {str(e)}")
        print("----------------------")
        print("="*50 + "\n")
        
        application_data["processing_status"] = f"Analysis Error: {str(e)}"
        return application_data