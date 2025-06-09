from pydantic import BaseModel, Field
from typing import Optional, Union

class LoanApplication(BaseModel):
    purpose_of_loan: Optional[str] = Field(default="NA", title="Purpose of loan")
    applicant_name: Optional[str] = Field(default="NA", title="Applicant Name")
    applicant_address: Optional[str] = Field(default="NA", title="Applicant Address")
    applicant_father_husband: Optional[str] = Field(default="NA", title="Applicant Name of father/husband")
    applicant_age: Optional[Union[str, int]] = Field(default="NA", title="Applicant Age")
    applicant_category: Optional[str] = Field(default="NA", title="Applicant Category")
    applicant_employment: Optional[str] = Field(default="NA", title="Applicant Employment institution name")
    applicant_retirement_date: Optional[str] = Field(default="NA", title="Applicant Retirement date")
    applicant_years_service: Optional[Union[str, int]] = Field(default="NA", title="Applicant Completed years of service")
    applicant_gross_salary: Optional[Union[str, int]] = Field(default="NA", title="Applicant Gross salary")
    applicant_other_incomes: Optional[str] = Field(default="NA", title="Applicant Other incomes")
    applicant_deposit_details: Optional[str] = Field(default="NA", title="Applicant Any Deposit details")
    applicant_loan_details: Optional[str] = Field(default="NA", title="Applicant Existing Loan details")
    applicant_landed_property: Optional[str] = Field(default="NA", title="Applicant Landed property details")
    
    spouse_name: Optional[str] = Field(default="NA", title="Spouse's Name")
    spouse_address: Optional[str] = Field(default="NA", title="Spouse's Address")
    spouse_father_husband: Optional[str] = Field(default="NA", title="Spouse's Name of father/husband")
    spouse_age: Optional[str] = Field(default="NA", title="Spouse's Age")
    spouse_category: Optional[str] = Field(default="NA", title="Spouse's Category")
    spouse_employment: Optional[str] = Field(default="NA", title="Spouse's Employment institution name")
    spouse_retirement_date: Optional[str] = Field(default="NA", title="Spouse's Retirement date")
    spouse_years_service: Optional[str] = Field(default="NA", title="Spouse's Completed years of service")
    spouse_gross_salary: Optional[str] = Field(default="NA", title="Spouse's Gross salary")
    spouse_other_incomes: Optional[str] = Field(default="NA", title="Spouse's Other incomes")
    spouse_deposit_details: Optional[str] = Field(default="NA", title="Spouse's Any Deposit details")
    spouse_loan_details: Optional[str] = Field(default="NA", title="Spouse's Existing Loan details")
    spouse_landed_property: Optional[str] = Field(default="NA", title="Spouse's Landed property details")
    
    guarantor1_name: Optional[str] = Field(default="NA", title="Guarantor1 name")
    guarantor1_address: Optional[str] = Field(default="NA", title="Guarantor1 permanent address")
    guarantor1_father_husband: Optional[str] = Field(default="NA", title="Guarantor1 father/husband names")
    guarantor1_occupations: Optional[str] = Field(default="NA", title="Guarantor1 occupations")
    guarantor1_salary: Optional[str] = Field(default="NA", title="Guarantor1 income from salary")
    guarantor1_other_incomes: Optional[str] = Field(default="NA", title="Guarantor1 income from other sources")
    guarantor1_assets: Optional[str] = Field(default="NA", title="Guarantor1 landed assets details")
    guarantor1_relationship: Optional[str] = Field(default="NA", title="Guarantor1 relationship with applicant")
    guarantor1_deposits: Optional[str] = Field(default="NA", title="Guarantor1 bank deposit details")
    guarantor1_loan_details: Optional[str] = Field(default="NA", title="Guarantor1 loan details")
    
    guarantor2_name: Optional[str] = Field(default="NA", title="Guarantor2 name")
    guarantor2_address: Optional[str] = Field(default="NA", title="Guarantor2 permanent address")
    guarantor2_father_husband: Optional[str] = Field(default="NA", title="Guarantor2 father/husband names")
    guarantor2_occupations: Optional[str] = Field(default="NA", title="Guarantor2 occupations")
    guarantor2_salary: Optional[str] = Field(default="NA", title="Guarantor2 income from salary")
    guarantor2_other_incomes: Optional[str] = Field(default="NA", title="Guarantor2 income from other sources")
    guarantor2_assets: Optional[str] = Field(default="NA", title="Guarantor2 landed assets details")
    guarantor2_relationship: Optional[str] = Field(default="NA", title="Guarantor2 relationship with applicant")
    guarantor2_deposits: Optional[str] = Field(default="NA", title="Guarantor2 bank deposit details")
    guarantor2_loan_details: Optional[str] = Field(default="NA", title="Guarantor2 loan details")
    
    investment_proposed: Optional[str] = Field(default="NA", title="Investment proposed")
    margin_offered: Optional[str] = Field(default="NA", title="Margin offered")
    bank_finance_required: Optional[str] = Field(default="NA", title="Bank finance required")
    repayment_period: Optional[str] = Field(default="NA", title="Repayment period required")
    collateral_details: Optional[str] = Field(default="NA", title="Collateral land/building details")
    lic_policy_details: Optional[str] = Field(default="NA", title="LIC policy details")
    deposit_details: Optional[str] = Field(default="NA", title="NSC/KVP/Bank/Post Office deposit details")
    date: Optional[str] = Field(default="NA", title="Date")
    
    # Method to convert snake_case to original field names
    def to_original_field_dict(self) -> dict:
        field_mapping = {
            "purpose_of_loan": "Purpose of loan",
            "applicant_name": "Applicant Name",
            "applicant_address": "Applicant Address",
            "applicant_father_husband": "Applicant Name of father/husband",
            "applicant_age": "Applicant Age",
            "applicant_category": "Applicant Category",
            "applicant_employment": "Applicant Employment institution name",
            "applicant_retirement_date": "Applicant Retirement date",
            "applicant_years_service": "Applicant Completed years of service",
            "applicant_gross_salary": "Applicant Gross salary",
            "applicant_other_incomes": "Applicant Other incomes",
            "applicant_deposit_details": "Applicant Any Deposit details",
            "applicant_loan_details": "Applicant Existing Loan details",
            "applicant_landed_property": "Applicant Landed property details",
            
            "spouse_name": "Spouse's Name",
            "spouse_address": "Spouse's Address",
            "spouse_father_husband": "Spouse's Name of father/husband",
            "spouse_age": "Spouse's Age",
            "spouse_category": "Spouse's Category",
            "spouse_employment": "Spouse's Employment institution name",
            "spouse_retirement_date": "Spouse's Retirement date",
            "spouse_years_service": "Spouse's Completed years of service",
            "spouse_gross_salary": "Spouse's Gross salary",
            "spouse_other_incomes": "Spouse's Other incomes",
            "spouse_deposit_details": "Spouse's Any Deposit details",
            "spouse_loan_details": "Spouse's Existing Loan details",
            "spouse_landed_property": "Spouse's Landed property details",
            
            "guarantor1_name": "Guarantor1 name",
            "guarantor1_address": "Guarantor1 permanent address",
            "guarantor1_father_husband": "Guarantor1 father/husband names",
            "guarantor1_occupations": "Guarantor1 occupations",
            "guarantor1_salary": "Guarantor1 income from salary",
            "guarantor1_other_incomes": "Guarantor1 income from other sources",
            "guarantor1_assets": "Guarantor1 landed assets details",
            "guarantor1_relationship": "Guarantor1 relationship with applicant",
            "guarantor1_deposits": "Guarantor1 bank deposit details",
            "guarantor1_loan_details": "Guarantor1 loan details",
            
            "guarantor2_name": "Guarantor2 name",
            "guarantor2_address": "Guarantor2 permanent address",
            "guarantor2_father_husband": "Guarantor2 father/husband names",
            "guarantor2_occupations": "Guarantor2 occupations",
            "guarantor2_salary": "Guarantor2 income from salary",
            "guarantor2_other_incomes": "Guarantor2 income from other sources",
            "guarantor2_assets": "Guarantor2 landed assets details",
            "guarantor2_relationship": "Guarantor2 relationship with applicant",
            "guarantor2_deposits": "Guarantor2 bank deposit details",
            "guarantor2_loan_details": "Guarantor2 loan details",
            
            "investment_proposed": "Investment proposed",
            "margin_offered": "Margin offered",
            "bank_finance_required": "Bank finance required",
            "repayment_period": "Repayment period required",
            "collateral_details": "Collateral land/building details",
            "lic_policy_details": "LIC policy details",
            "deposit_details": "NSC/KVP/Bank/Post Office deposit details",
            "date": "Date"
        }
        
        return {field_mapping[k]: v for k, v in self.dict().items()}