from typing import Dict, List, Tuple

# List of fields that must be integers
INT_FIELDS = [
    "Applicant Age",
    "Applicant Gross salary", 
    "Applicant Other incomes",
    "Guarantor1 income from salary",
    "Guarantor2 income from salary",
    "Investment proposed",
    "Margin offered",
    "Bank finance required"
]

# Define the list of required fields that must have values
REQUIRED_FIELDS = [
    "Purpose of loan",
    "Applicant Name",
    "Applicant Address",
    "Applicant Name of father/husband",
    "Applicant Age",
    "Applicant Employment institution name",
    "Applicant Retirement date",
    "Applicant Completed years of service",
    "Applicant Gross salary",
    "Applicant Landed property details",
    "Guarantor1 name",
    "Guarantor1 permanent address",
    "Guarantor1 father/husband names",
    "Guarantor1 occupations",
    "Guarantor1 income from salary",
    "Guarantor1 landed assets details",
    "Guarantor1 relationship with applicant",
    "Investment proposed",
    "Margin offered",
    "Bank finance required",
    "Repayment period required",
    "Collateral land/building details",
    "Date"
]

def validate_application(application_data: Dict, field_keys: List[str]) -> Tuple[bool, List[str], List[str], List[str]]:
    """
    Validate an application by checking for required fields, integer validation, and financial structure.
    Only specified required fields must be present, other fields can be NA.
    
    Args:
        application_data (dict): Dictionary containing application data
        field_keys (list): List of all field names that should exist
        
    Returns:
        Tuple[bool, List[str], List[str], List[str]]: 
            (is_valid, list_of_missing_fields, list_of_invalid_type_fields, list_of_invalid_financial_structure)
    """
    missing_fields = []
    invalid_type_fields = []
    invalid_financial_structure = []
    
    # Check only required fields for missing values
    for field in REQUIRED_FIELDS:
        # Check if required field is missing, empty string, or has "NA" value
        if (field not in application_data or 
            application_data[field] == "" or 
            application_data[field] == "NA"):
            missing_fields.append(field)
    
    # Check all fields for integer validation, regardless if required or not
    for field in field_keys:
        if field in INT_FIELDS and field in application_data:
            try:
                # Only validate if the field has a value other than "NA"
                value = application_data[field]
                if value != "NA" and isinstance(value, str):
                    # Check if the string contains only digits
                    if not value.isdigit():
                        invalid_type_fields.append(field)
                        continue
                    int(value)
            except (ValueError, TypeError):
                invalid_type_fields.append(field)
    
    # Validate financial structure
    # Check if all required fields for financial validation exist and are numeric
    investment = application_data.get("Investment proposed", "NA")
    margin = application_data.get("Margin offered", "NA")
    bank_finance = application_data.get("Bank finance required", "NA")
    
    if all(field != "NA" for field in [investment, margin, bank_finance]):
        try:
            # Print values for debugging
            print(f"Validating financial structure: Investment={investment}, Margin={margin}, Bank Finance={bank_finance}")
            
            investment_val = int(investment)
            margin_val = int(margin)
            bank_finance_val = int(bank_finance)
            
            # Debug calculations
            funding_gap = investment_val - margin_val
            print(f"Calculated funding gap: {funding_gap}")
            
            # Validate: Bank finance <= (Investment - Margin)
            if bank_finance_val > (investment_val - margin_val):
                print(f"VALIDATION FAILED: Bank finance {bank_finance_val} exceeds funding gap {funding_gap}")
                invalid_financial_structure.append("Bank finance exceeds funding gap")

            # Validate: Margin <= Investment
            if margin_val > investment_val:
                print(f"VALIDATION FAILED: Margin {margin_val} exceeds investment {investment_val}")
                invalid_financial_structure.append("Margin amount exceeds total investment amount")
                
        except (ValueError, TypeError) as e:
            print(f"Error validating financial structure: {e}")
            # If any conversion failed but wasn't caught by the integer validation
            pass
            
        # Debug the validation results
        print(f"Financial structure validation results: {invalid_financial_structure}")
    
    # Application is valid if there are no validation issues
    is_valid = (len(missing_fields) == 0 and 
                len(invalid_type_fields) == 0 and 
                len(invalid_financial_structure) == 0)
    
    return is_valid, missing_fields, invalid_type_fields, invalid_financial_structure

def process_application(application_data: Dict, field_keys: List[str]) -> Dict:
    """
    Process a single application and check for missing required fields.
    
    Args:
        application_data (dict): Dictionary containing application data
        field_keys (list): List of field names that should be validated
        
    Returns:
        dict: Processing results with validation status
    """
    # Get application identifier
    app_id = application_data.get("Application", "UNKNOWN")
    
    # Validate the application
    is_valid, missing_fields, invalid_type_fields, invalid_financial_structure = validate_application(application_data, field_keys)
    
    # Create result object
    result = {
        "Application": app_id,
        "has_missing_fields": len(missing_fields) > 0,
        "missing_fields": missing_fields,
        "has_invalid_types": len(invalid_type_fields) > 0,
        "invalid_type_fields": invalid_type_fields,
        "has_invalid_financial_structure": len(invalid_financial_structure) > 0,
        "invalid_financial_structure": invalid_financial_structure,
        "application_data": application_data,
        "status": "Complete" if is_valid else "Incomplete"
    }
    
    return result