from typing import Dict, List
import logging
from colorama import init, Fore, Style

init()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

def print_step_header(step: str):
    logging.info(f"\n=== {step} ===")

def print_record_status(current: int, total: int, id_info: str = ""):
    logging.info(f"[{current}/{total}] Processing{' ' + id_info if id_info else ''}")

def print_financial_summary(rating: str, strengths: List[str], concerns: List[str]):
    logging.info(f"Rating: {rating}")
    if strengths:
        logging.info(f"Key Strength: {strengths[0]}")
    if concerns:
        logging.info(f"Key Concern: {concerns[0]}")

def print_risk_summary(rating: str, major_risks: List[str], mitigating_factors: List[str]):
    logging.info(f"Risk: {rating}")
    if major_risks:
        logging.info(f"Major Risk: {major_risks[0]}")
    if mitigating_factors:
        logging.info(f"Mitigating: {mitigating_factors[0]}")

def print_decision_summary(decision: Dict):
    logging.info(f"Decision: {decision['decision']}")
    logging.info(f"Confidence: {decision['confidence_level']}")
    if decision["key_decision_factors"]:
        logging.info(f"Key Factor: {decision['key_decision_factors'][0]}")

def print_completion_summary(file_path: str, count: int):
    logging.info(f"\n✓ Processed {count} records")
    logging.info(f"Output: {file_path}")