import os
import io
import asyncio
import aiohttp
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Optional,Literal
from fastapi import FastAPI, File, UploadFile, Query, HTTPException,APIRouter
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from DigitalMarketing.utils import (
    validate_file,
    perform_segmentation,
    generate_chart_data,
    generate_recommendations,
    get_segment_products,
    generate_email_content,
    generate_personalized_template,
)
from DocBot.docbot import upload_documents

router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
for dir_path in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)
    
data_store: Dict[str, pd.DataFrame] = {}
uploaded_files: Dict[str, bool] = {
    'customers': False,
    'products': False,
    'purchases': False
}
cross_sell_recs = []
upsell_recs = []

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ['.csv']
TOKEN_OPTIONS = [100, 500, 1000, 2000, 4000]

router.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

@router.post("/upload/{file_type}")
async def upload_file(file_type: str, file: UploadFile = File(...)):
    """Upload and validate data files"""
    try:
        validate_file(file)
        file_path = os.path.join(UPLOAD_DIR, f"{file_type}.csv")
        
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            df = pd.read_csv(file_path)
            data_store[file_type] = df
            uploaded_files[file_type] = True

            return {
                "status": "success",
                "message": f"{file_type} data uploaded successfully",
                }
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="The uploaded file is empty")
        except pd.errors.ParserError:
            raise HTTPException(status_code=400, detail="Invalid CSV format")
            
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/process")
async def process_data():
    """Process uploaded data and generate insights"""
    try:
        if not all(uploaded_files.values()):
            raise HTTPException(
                status_code=400, 
                detail="Please upload all required files first"
            )
        
        # 1. Customer Segmentation
        segmented_customers, cluster_profiles = perform_segmentation(data_store['customers'])
        
        # 2. Generate chart data
        chart_data = generate_chart_data(segmented_customers)
        
        # 3. Generate Recommendations
        cross_sell, up_sell = generate_recommendations(
            segmented_customers, 
            data_store['products'], 
            data_store['purchases']
        )
        
        # Save outputs
        outputs = {
            'segmented_customers': 'segmented_customers.csv',
            'cluster_profiles': 'cluster_profiles.csv',
            'cross_sell': 'cross_sell.csv',
            'up_sell': 'up_sell.csv'
        }

        saved_files = {}
        for name, filename in outputs.items():
            df = locals()[name]
            if isinstance(df, pd.DataFrame):
                file_path = os.path.join(OUTPUT_DIR, filename)
                df.to_csv(file_path, index=False)
                saved_files[name] = file_path

        # Collect files from UPLOAD_DIR
        additional_files = [
            os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))
        ]

        # Combine all files for upload
        all_files = list(saved_files.values()) + additional_files

        # Send all files to the document upload API in a single request
        asyncio.create_task(upload_files_to_api(all_files, username="digital_marketing"))
        
        return {
            "message": "Data processed successfully",
            "chart_data": chart_data,
            "outputs": outputs
        }
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def upload_files_to_api(file_paths, username):
    """Directly call the upload_documents function instead of making an API request."""
    try:
        uploaded_files = []
        
        # Convert file paths into UploadFile objects
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                file_obj = UploadFile(
                    filename=os.path.basename(file_path),
                    file=io.BytesIO(file_bytes),
                )
                uploaded_files.append(file_obj)

        # Directly call the FastAPI function
        upload_result = await upload_documents(uploaded_files, username)

        return {
            "status_code": 200,
            "response": upload_result
        }

    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        return {"status_code": 500, "error": str(e)}

@router.get("/available-customers")
async def get_available_customers(
    recommendation_type: Literal["cross_sell", "up_sell"],
    segment: str
):
    """Get available customers for a segment"""
    try:
        recommendations_file = os.path.join(OUTPUT_DIR, f'{recommendation_type}.csv')
        if not os.path.exists(recommendations_file):
            raise HTTPException(
                status_code=404,
                detail="Recommendations file not found. Please process data first."
            )

        recommendations_df = pd.read_csv(recommendations_file)
        segment_customers = recommendations_df[
            recommendations_df['segment_label'] == segment
        ]['customer_id'].unique().tolist()
        
        if not segment_customers:
            return {
                "status": "success",
                "segment": segment,
                "customers": [],
                "message": "No customers found for this segment"
            }
        
        return {
            "status": "success",
            "segment": segment,
            "customers": segment_customers
        }
        
    except Exception as e:
        logger.error(f"Error getting available customers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-personalized-emails")
async def generate_personalized_emails(
    recommendation_type: Literal["cross_sell", "up_sell"],
    segment: str,
    max_tokens: int = Query(default=1000, ge=100, le=4000),
    customer_ids: List[str] = Query(None)
):
    """Generate personalized email templates for customers"""
    try:
        # Input validation
        if not customer_ids:
            raise HTTPException(status_code=400, detail="No customer IDs provided")

        if max_tokens not in TOKEN_OPTIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid token value. Allowed values: {TOKEN_OPTIONS}"
            )

        recommendations_file = os.path.join(OUTPUT_DIR, f'{recommendation_type}.csv')
        if not os.path.exists(recommendations_file):
            raise HTTPException(
                status_code=404, 
                detail="Recommendations file not found. Please process data first."
            )

        recommendations_df = pd.read_csv(recommendations_file)
        generated_templates = []
        
        async def process_customer(customer_id: str) -> Optional[Dict]:
            try:
                customer_recs = recommendations_df[
                    (recommendations_df['customer_id'] == customer_id) &
                    (recommendations_df['segment_label'] == segment)
                ]
                
                if customer_recs.empty:
                    logger.warning(f"No recommendations found for customer {customer_id}")
                    return None
                
                products_by_category = get_segment_products(customer_recs, segment)
                
                email_content = await generate_email_content(
                    customer_id,
                    segment,
                    products_by_category,
                    recommendation_type,
                    max_tokens
                )
                
                html_content = generate_personalized_template(
                    customer_id,
                    segment,
                    products_by_category,
                    recommendation_type,
                    email_content
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"email_template_{customer_id}_{segment.lower().replace(' ', '_')}_{recommendation_type}_{timestamp}.html"
                
                with open(os.path.join(OUTPUT_DIR, filename), 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                return {
                    "customer_id": customer_id,
                    "segment": segment,
                    "template_file": filename,
                    "subject_line": email_content['subject_line'],
                    "product_count": sum(len(products) for products in products_by_category.values())
                }

            except Exception as e:
                logger.error(f"Error processing customer {customer_id}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None

        # Process customers concurrently
        tasks = [process_customer(cid) for cid in customer_ids]
        results = await asyncio.gather(*tasks)
        
        # Filter out failed generations
        generated_templates = [r for r in results if r is not None]
        
        if not generated_templates:
            logger.warning("No templates were generated successfully")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any email templates"
            )

        logger.info(f"Successfully generated {len(generated_templates)} templates")
        return {
            "message": "Personalized email templates generated successfully",
            "templates_generated": len(generated_templates),
            "templates": generated_templates
        }
        
    except Exception as e:
        logger.error(f"Error in generate_personalized_emails: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/static/{filename}")
async def get_static_file(filename: str):
    """Serve static files"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

async def cleanup_old_files():
    """Clean up files older than 24 hours"""
    while True:
        try:
            current_time = datetime.now()
            for dir_path in [UPLOAD_DIR, OUTPUT_DIR]:
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if (current_time - file_time).days >= 1:
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
        await asyncio.sleep(3600)  # Run every hour

@router.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(cleanup_old_files())