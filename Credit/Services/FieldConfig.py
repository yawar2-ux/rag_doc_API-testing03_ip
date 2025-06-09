import os

# Base paths - all other paths should be derived from these
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Ensure directories exist
for directory in [TEMP_DIR, OUTPUT_DIR, UPLOAD_DIR]:
    os.makedirs(directory, exist_ok=True)

# Standard CSV filename for all processing results
STANDARD_CSV_FILENAME = "loan_applications_data.csv"
STANDARD_CSV_PATH = os.path.join(OUTPUT_DIR, STANDARD_CSV_FILENAME)

keys = [
        "Purpose of loan", "Applicant Name", "Applicant Address",
        "Applicant Name of father/husband", "Applicant Age", "Applicant Category",
        "Applicant Employment institution name", "Applicant Retirement date",
        "Applicant Completed years of service", "Applicant Gross salary",
        "Applicant Other incomes", "Applicant Any Deposit details",
        "Applicant Existing Loan details", "Applicant Landed property details",
        "Spouse's Name", "Spouse's Address", "Spouse's Name of father/husband",
        "Spouse's Age", "Spouse's Category", "Spouse's Employment institution name",
        "Spouse's Retirement date", "Spouse's Completed years of service",
        "Spouse's Gross salary", "Spouse's Other incomes", "Spouse's Any Deposit details",
        "Spouse's Existing Loan details", "Spouse's Landed property details",
        "Guarantor1 name", "Guarantor1 permanent address", "Guarantor1 father/husband names",
        "Guarantor1 occupations", "Guarantor1 income from salary",
        "Guarantor1 income from other sources", "Guarantor1 landed assets details",
        "Guarantor1 relationship with applicant", "Guarantor1 bank deposit details",
        "Guarantor1 loan details", "Guarantor2 name", "Guarantor2 permanent address",
        "Guarantor2 father/husband names", "Guarantor2 occupations",
        "Guarantor2 income from salary", "Guarantor2 income from other sources",
        "Guarantor2 landed assets details", "Guarantor2 relationship with applicant",
        "Guarantor2 bank deposit details", "Guarantor2 loan details",
        "Investment proposed", "Margin offered", "Bank finance required",
        "Repayment period required", "Collateral land/building details",
        "LIC policy details", "NSC/KVP/Bank/Post Office deposit details", "Date"
    ]