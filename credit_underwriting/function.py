import asyncio
import json
import os
from dotenv import load_dotenv

from fastapi import HTTPException
from credit_underwriting.decision_agent import DecisionAgent
from credit_underwriting.document_agent import DocumentAgent
from credit_underwriting.financial_agent import FinancialAgent
from credit_underwriting.risk_agent import RiskAgent

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY_DC")

async def run_agent_task(agent_name, task_func):
    """Wrapper to run agent tasks asynchronously"""
    yield f"data: {json.dumps({'status': 'processing', 'agent': agent_name})}\n\n"
    await asyncio.sleep(0.1)  # Small delay for frontend to catch up

    try:
        result = await asyncio.to_thread(task_func)
        yield f"data: {json.dumps({'status': 'completed', 'agent': agent_name})}\n\n"
        # Pass result back through the next yield instead of return
        yield f"data: {json.dumps({'status': 'result', 'agent': agent_name, 'data': result})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'status': 'error', 'agent': agent_name, 'message': str(e)})}\n\n"
        raise e

async def process_loan_application(file_path):
    # Initialize all file variables to None at the start
    financial_file = None
    risk_file = None
    log_file = None
    missing_data = {}
    financial_analysis_file = None
    risk_analysis_file = None
    final_decisions_file = None

    files_to_cleanup = [file_path]  # Start with the input file

    try:
        # Document Processing
        doc_agent = DocumentAgent()
        doc_result = None
        async for msg in run_agent_task('document', lambda: doc_agent.process_documents(file_path)):
            if msg.startswith('data: '):
                data = json.loads(msg[6:])
                if data.get('status') == 'result':
                    doc_result = data.get('data')
                yield msg

        if not doc_result or not isinstance(doc_result, list) or len(doc_result) < 3:
            raise HTTPException(status_code=500, detail="Document processing failed")
        
        financial_file, risk_file, log_file, missing_data = doc_result
        files_to_cleanup.extend([financial_file, risk_file, log_file])
        
        # Send missing data information
        if missing_data:
            missing_data_list = [{"customer_id": k, "missing_fields": v} for k, v in missing_data.items()]
            yield f"data: {json.dumps({'status': 'missing_data', 'data': missing_data_list})}\n\n"
        
        # Check if we have valid records to process
        try:
            with open(financial_file, 'r') as f:
                has_data = len(f.readlines()) > 1  # Check if there's more than just the header
        except:
            has_data = False
            
        if not has_data:
            yield f"data: {json.dumps({'status': 'completed', 'message': 'No valid records to process'})}\n\n"
            return

        # Financial Analysis
        financial_agent = FinancialAgent(GROQ_API_KEY)
        financial_result = None
        async for msg in run_agent_task('financial', lambda: financial_agent.process_records(financial_file)):
            if msg.startswith('data: '):
                data = json.loads(msg[6:])
                if data.get('status') == 'result':
                    financial_result = data.get('data')
                yield msg

        if not financial_result:
            raise HTTPException(status_code=500, detail="Financial analysis failed")
        financial_analysis_file = financial_result
        files_to_cleanup.append(financial_analysis_file)

        # Risk Analysis
        risk_agent = RiskAgent(GROQ_API_KEY)
        risk_result = None
        async for msg in run_agent_task('risk', lambda: risk_agent.process_records(risk_file)):
            if msg.startswith('data: '):
                data = json.loads(msg[6:])
                if data.get('status') == 'result':
                    risk_result = data.get('data')
                yield msg

        if not risk_result:
            raise HTTPException(status_code=500, detail="Risk analysis failed")
        risk_analysis_file = risk_result
        files_to_cleanup.append(risk_analysis_file)

        # Final Decision
        decision_agent = DecisionAgent(GROQ_API_KEY)
        final_result = None
        async for msg in run_agent_task('decision',
            lambda: decision_agent.process_applications(financial_analysis_file, risk_analysis_file)):
            if msg.startswith('data: '):
                data = json.loads(msg[6:])
                if data.get('status') == 'result':
                    final_result = data.get('data')
                yield msg

        if not final_result:
            raise HTTPException(status_code=500, detail="Decision making failed")
        final_decisions_file = final_result
        files_to_cleanup.append(final_decisions_file)

        # Read and send final results
        with open(final_decisions_file, 'r') as f:
            results = f.read()

        yield f"data: {json.dumps({'status': 'completed', 'agent': 'decision', 'results': results})}\n\n"

    except Exception as e:
        error_msg = str(e)
        yield f"data: {json.dumps({'status': 'error', 'message': error_msg})}\n\n"
        raise HTTPException(status_code=500, detail=error_msg)

    finally:
        # Clean up only the files that were actually created
        for file in files_to_cleanup:
            try:
                if file and os.path.exists(file):
                    os.remove(file)
            except Exception as cleanup_error:
                print(f"Failed to remove file {file}: {cleanup_error}")