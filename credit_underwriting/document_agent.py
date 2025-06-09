import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple
from credit_underwriting.utils import print_step_header, print_record_status, print_completion_summary

class DocumentAgent:
    def __init__(self):
        # Define validation rules for numeric fields
        self.numeric_validations = {
            'Annual Income': {'min': 0},
            'Net Monthly Income': {'min': 0},
            'Monthly Fixed Expenses': {'min': 0},
            'Debt': {'min': 0},
            'Loan Amount': {'min': 0},
            'Credit Utilization Ratio': {'min': 0, 'max': 1},
            'DTI': {'min': 0, 'max': 1},
            'Credit Score': {'min': 300, 'max': 900}
        }

        # Define valid categorical values
        self.categorical_values = {
            'Saving Account': ['Yes', 'No'],
            'Owns A House': ['Yes', 'No'],
            'Insurance Coverage': ['Yes', 'No'],
            'Insurance Interest': ['Yes', 'No'],
            'Loan Type': ['Personal', 'Home', 'Vehicle', 'Education', 'Business'],
            'Constitution': ['Resident Individual', 'Non Resident Individual', 'Foreign National'],
            'Marital Status': ['Single', 'Married', 'Divorced', 'Widowed'],
            'Education Level': ['Graduate', 'Post Graduate', 'Doctorate', 'High School'],
            'Loan Repayment History': ['Excellent', 'Good', 'Fair', 'Poor'],
            'Litigation History': ['Clear', 'Civil Case'],
            'Aadhaar Present Address': ['Yes', 'No'],
            'Aadhaar Permanent Address': ['Yes', 'No']
        }

        # Define required fields that cannot be empty
        self.required_fields = [
            'Customer ID', 'Name (As Per ID)', 'Name (On Card)', 'Mobile Number', 
            'Phone Number', 'Email ID', 'Loan Amount', 'Annual Income', 
            'Net Monthly Income', 'Credit Score'
        ]

        # Define columns for financial and risk records
        self.financial_columns = [
            'Customer ID', 'Saving Account', 'City', 'Constitution',
            'Annual Income', 'Net Monthly Income', 'Owns A House',
            'Work Experience', 'Marital Status', 'Credit Utilization Ratio',
            'Monthly Fixed Expenses', 'Education Level', 'Insurance Coverage'
        ]

        self.risk_columns = [
            'Customer ID', 'Debt', 'DTI', 'Credit Score', 'Dependents',
            'Employement Type', 'Loan Type', 'City', 'Constitution',
            'Aadhaar Present Address', 'Aadhaar Permanent Address',
            'Annual Income', 'Net Monthly Income', 'Loan Purpose',
            'Loan Amount', 'Reference 1 Relationship', 'Reference 1 City',
            'Insurance Interest', 'Loan Repayment History', 'Litigation History'
        ]

    def validate_record(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """Validate a single record and return lists of issues and missing values"""
        issues = []
        missing_values = []
        customer_id = row['Customer ID']

        # Check for missing required values (record will be skipped if any of these are missing)
        for field in self.required_fields:
            if field in row:
                if pd.isna(row[field]) or str(row[field]).strip() == '':
                    missing_values.append(f"{field}")
        
        # Check all fields for missing values (for reporting purposes)
        all_missing_fields = []
        for field, value in row.items():
            if pd.isna(value) or str(value).strip() == '':
                all_missing_fields.append(field)
        
        # Store all missing fields in the customer record (including non-required fields)
        if all_missing_fields:
            # Replace missing_values with all_missing_fields if you want to include all missing fields
            # Otherwise, keep missing_values as is to only skip records with missing required fields
            missing_values = all_missing_fields

        # Validate numeric fields
        for field, rules in self.numeric_validations.items():
            if field in row:
                try:
                    if pd.isna(row[field]) or str(row[field]).strip() == '':
                        continue  # Skip validation for missing values, they're handled above
                    value = float(row[field])
                    if 'min' in rules and value < rules['min']:
                        issues.append(f"Cust ID {customer_id}: {field} ({value}) below minimum value {rules['min']}")
                    if 'max' in rules and value > rules['max']:
                        issues.append(f"Cust ID {customer_id}: {field} ({value}) above maximum value {rules['max']}")
                except ValueError:
                    issues.append(f"Cust ID {customer_id}: Invalid {field} value - must be numeric")

        # Validate categorical fields
        for field, valid_values in self.categorical_values.items():
            if field in row and not pd.isna(row[field]) and row[field] not in valid_values:
                issues.append(f"Cust ID {customer_id}: Invalid {field} value '{row[field]}' - must be one of {valid_values}")

        return issues, missing_values

    def process_documents(self, input_file: str) -> Tuple[str, str, str, Dict]:
        """Process input documents and create financial and risk CSVs"""
        try:
            print_step_header("Document Processing")

            df = pd.read_csv(input_file)
            print(f"Found {len(df)} records")

            all_issues = []
            valid_records = []
            records_with_missing_values = {}

            for idx, row in enumerate(df.iterrows(), 1):
                customer_id = str(row[1].get('Customer ID', f'Unknown-{idx}'))
                print_record_status(idx, len(df), f"Customer {customer_id}")
                
                issues, missing_values = self.validate_record(row[1])
                
                if issues:
                    all_issues.extend(issues)
                
                if missing_values:
                    records_with_missing_values[customer_id] = missing_values
                    print(f"Skipping customer {customer_id} due to missing values: {', '.join(missing_values)}")
                elif not issues:
                    valid_records.append(row[1])

            valid_df = pd.DataFrame(valid_records)
            financial_df = valid_df[self.financial_columns].copy() if not valid_df.empty else pd.DataFrame(columns=self.financial_columns)
            risk_df = valid_df[self.risk_columns].copy() if not valid_df.empty else pd.DataFrame(columns=self.risk_columns)

            # Save files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            financial_file = 'financial_records.csv'
            risk_file = 'risk_factors.csv'
            # store logs in log folder, create folder if it doesn't exist
            log_folder = 'logs'
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            log_file = os.path.join(log_folder, f'validation_log_{timestamp}.txt')
            financial_df.to_csv(financial_file, index=False)
            risk_df.to_csv(risk_file, index=False)

            with open(log_file, 'w') as f:
                f.write("=== DATA VALIDATION REPORT ===\n\n")
                
                # Log missing values
                if records_with_missing_values:
                    f.write("MISSING VALUES:\n")
                    for cust_id, missing in records_with_missing_values.items():
                        f.write(f"Customer ID {cust_id}: Missing {', '.join(missing)}\n")
                    f.write("\n")
                
                # Log other validation issues
                if all_issues:
                    f.write("VALIDATION ISSUES:\n")
                    f.write('\n'.join(all_issues) + '\n\n')
                
                f.write(f"Total records: {len(df)}\n")
                f.write(f"Valid: {len(valid_records)}\n")
                f.write(f"With missing values: {len(records_with_missing_values)}\n")
                f.write(f"With other issues: {len(all_issues)}\n")

            print_completion_summary(financial_file, len(valid_records))
            return financial_file, risk_file, log_file, records_with_missing_values

        except Exception as e:
            print(f"Error: {str(e)}")
            return None, None, None, {}