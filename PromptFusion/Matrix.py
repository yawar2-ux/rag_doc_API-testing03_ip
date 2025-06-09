from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from bert_score import score

router = APIRouter()

class MatrixRequest(BaseModel):
    ground_truth: str
    model_responses: Dict[str, str]  # Format: {"model_name": "response"}

class ScoreResult(BaseModel):
    model_name: str
    precision: float
    recall: float
    f1: float

class MatrixResponse(BaseModel):
    results: List[ScoreResult]
    best_model: str  # Model with highest F1 score

@router.post("/evaluate", response_model=MatrixResponse)
async def evaluate_responses(request: MatrixRequest):
    """
    Evaluate model responses against ground truth using BERTScore
    """
    results = []
    best_f1 = -1
    best_model = ""
    
    # Get all model responses in a list
    responses = []
    model_names = []
    
    for model_name, response in request.model_responses.items():
        responses.append(response)
        model_names.append(model_name)
    
    if len(responses) == 0:
        return MatrixResponse(results=[], best_model="")
    
    # Calculate BERTScore for all responses against ground truth
    # References is repeated for each candidate
    references = [request.ground_truth] * len(responses) 
    
    # Get P, R, F1 scores
    P, R, F1 = score(
        cands=responses,
        refs=references,
        lang="en",
        verbose=False
    )
    
    # Convert to Python floats
    P = [float(p) for p in P]
    R = [float(r) for r in R]
    F1 = [float(f) for f in F1]
    
    # Create results
    for i, model_name in enumerate(model_names):
        result = ScoreResult(
            model_name=model_name,
            precision=P[i],
            recall=R[i],
            f1=F1[i]
        )
        results.append(result)
        
        # Track best model
        if F1[i] > best_f1:
            best_f1 = F1[i]
            best_model = model_name
    
    return MatrixResponse(
        results=results,
        best_model=best_model
    )