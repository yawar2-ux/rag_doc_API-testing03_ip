import io
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile
import warnings
import urllib3
import os
from io import StringIO


from DataEngineering.MissingValues import (
     generate_column_stats, generate_profiling_report, generate_smart_prompt, query_ollama
)

from DataEngineering.OutlinerFunctions import (
detect_outliers,
    process_outliers
    )

from DataEngineering.SyntheticFunction  import SyntheticDataOllama
generator = SyntheticDataOllama()



from DataEngineering.SyntheticTestCaseFunction import GenericCSVTestGenerator
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

testGenerator = GenericCSVTestGenerator()

print(os.getenv('OLLAMA_TEXT_MODEL'))
print(os.getenv('OLLAMA_BASE_URL'))


# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

router = APIRouter()

def clean_for_json(obj):
    """Clean data structures for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.where(pd.notnull(obj), None).to_dict()
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    return obj

@router.post("/missingValues")
async def clean_dataset(file: UploadFile):
    """
    API endpoint to clean a dataset using strict rules for missing value handling.
    """
    try:
        # Save uploaded file temporarily
        temp_file = NamedTemporaryFile(delete=False, suffix=".csv")
        with temp_file as f:
            f.write(await file.read())
        file_path = temp_file.name

        # Load dataset
        df_original = pd.read_csv(
            file_path,
            na_values=['', 'n', 'N', 'nan', 'NaN', 'NULL', 'null', 'None', 'NONE', 'NA', 'n/a', 'N/A', '#N/A','NAN'],
            parse_dates=True
        )
        print(f"Dataset Shape: {df_original.shape}")

        # Create a copy of the original DataFrame for cleaning
        # df_original_2=df_original.copy()
        # print(df_original_2.head(15))
        df_cleaned = df_original.copy()

        # Identify columns with missing values
        missing_columns = df_cleaned.columns[df_cleaned.isnull().any()].tolist()
        print(f"Found {len(missing_columns)} columns with missing values")

        # Process each column with missing values using Ollama
        for column in missing_columns:
            print(f"Processing column: {column}")
            column_stats = generate_column_stats(df_cleaned, column)
            prompt = generate_smart_prompt(column_stats)

            # Retry mechanism for Ollama query
            retries = 3
            for attempt in range(retries):
                try:
                    suggested_values = query_ollama(prompt)
                    if suggested_values:
                        if column_stats['type'] == 'numeric':
                            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mean())
                            df_cleaned[column] = df_cleaned[column].round(2)  # Round to 2 decimal places
                        elif column_stats['type'] == 'categorical':
                            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mode()[0])
                        elif column_stats['type'] == 'datetime':
                            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
                        print(f"Filled missing values in column '{column}' using Ollama suggestions.")
                        break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for column '{column}': {e}")
                    if attempt == retries - 1:
                        print(f"Failed to process column '{column}' after {retries} attempts. Skipping.")

        # Final cleanup for df_cleaned
        df_cleaned.replace([np.inf, -np.inf], None, inplace=True)
        df_cleaned = df_cleaned.where(pd.notnull(df_cleaned), None)  # Replace NaN with None for JSON compatibility

        # Final cleanup for df_original
        df_original.replace([np.inf, -np.inf, np.nan], None, inplace=True)
        df_original = df_original.where(pd.notnull(df_original), None)  # Replace NaN with None for JSON compatibility
        # print("---------------------------------------------")
        # print(df_original.head(15))
        # Generate profiling report
        html_report = generate_profiling_report(df_cleaned)

        # Save cleaned DataFrame to CSV
        try:
            cleaned_csv_path = "cleaned_df_final.csv"
            df_cleaned.to_csv(cleaned_csv_path, index=False)
            print(f"Cleaned DataFrame saved to {cleaned_csv_path}")
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")

        # Prepare JSON response
        response = {
            "message": "Data cleaned successfully",
            "original_df": df_original.astype(object).to_dict(orient="records"),  # Ensure JSON compliance
            "cleaned_df": df_cleaned.to_dict(orient="records"),
            "html_report": html_report
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Cleanup temporary file
        if 'temp_file' in locals():
            os.unlink(temp_file.name)

@router.post("/remove-outliers")
async def remove_outliers(file: UploadFile):
    """
    Endpoint to detect and remove outliers using Ollama decisions.
    """
    # Configuration
    model = os.getenv('OLLAMA_TEXT_MODEL')
    max_retries = 3
    retry_delay = 5

    try:
        # Save uploaded file temporarily
        temp_file = NamedTemporaryFile(delete=False, suffix=".csv")
        with temp_file as f:
            f.write(await file.read())
        file_path = temp_file.name

        # Load dataset
        df = pd.read_csv(file_path)
        # print(f"Dataset Shape: {df.shape}")

        # Get initial outlier percentages
        initial_outliers = detect_outliers(df)

        # Process outliers for each numerical column
        cleaned_df = df.copy()
        rows_removed = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            if initial_outliers.get(column, 0) > 0:
                initial_rows = len(cleaned_df)
                cleaned_df = process_outliers(
                    cleaned_df,
                    column,
                    model,
                    max_retries,
                    retry_delay
                )
                rows_removed[column] = initial_rows - len(cleaned_df)

        # Get final outlier percentages
        final_outliers = detect_outliers(cleaned_df)

        # Save cleaned DataFrame to CSV
        try:
            cleaned_df.to_csv("cleaned_outliers.csv", index=False)
            # print("Saved cleaned data to CSV")
        except Exception as e:
            print(f"Error saving CSV: {str(e)}")


        # Prepare response statistics
        response = {
            "original_data": df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None).to_dict(orient="records"),
            "cleaned_data": cleaned_df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(cleaned_df), None).to_dict(orient="records"),
            "outlier_statistics": {
                "initial_outliers": clean_for_json(initial_outliers),
                "final_outliers": clean_for_json(final_outliers),
            },
            "dataset_statistics": {
                "original_shape": list(df.shape),  # Convert tuple to list
                "cleaned_shape": list(cleaned_df.shape),
                "reduction_percentage": clean_for_json(
                    ((df.shape[0] - cleaned_df.shape[0]) / df.shape[0]) * 100
                )
            },
            "column_statistics": clean_for_json({
                col: {
                    "original": {
                        "mean": df[col].mean(),
                        "median": df[col].median(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max()
                    },
                    "cleaned": {
                        "mean": cleaned_df[col].mean(),
                        "median": cleaned_df[col].median(),
                        "std": cleaned_df[col].std(),
                        "min": cleaned_df[col].min(),
                        "max": cleaned_df[col].max()
                    }
                }
                for col in df.select_dtypes(include=[np.number]).columns
            }),
            "sample_rows": cleaned_df.head(5).replace([np.inf, -np.inf], np.nan).where(pd.notnull(cleaned_df.head(5)), None).to_dict(orient="records")
        }

        # Clean the entire response
        cleaned_response = clean_for_json(response)
        # print(cleaned_response)
        return JSONResponse(content=cleaned_response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Cleanup temporary file
        if 'temp_file' in locals():
            os.unlink(temp_file.name)


@router.post("/syntheticDataGenerate")
async def generate_synthetic_data(file: UploadFile = File(...), num_samples: int = Form(...)):
    """Endpoint to generate synthetic data."""
    try:
        # Read uploaded file into a Pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Validate input
        if num_samples <= 0:
            raise HTTPException(status_code=400, detail="Number of samples must be greater than 0")

        # Generate synthetic data
        synthetic_df = generator.generate_synthetic_data(df, num_samples)

        # Get DataFrame shapes
        original_shape = df.shape
        synthetic_shape = synthetic_df.shape

        # Compute statistics (mean, std, min, max) for numerical columns
        original_stats = df.describe().to_dict()
        synthetic_stats = synthetic_df.describe().to_dict()

        return JSONResponse(content={
            "message": "Synthetic data generated successfully.",
            "original_data_shape": original_shape,
            "synthetic_data_shape": synthetic_shape,
            "original_statistics": original_stats,
            "synthetic_statistics": synthetic_stats,
            "synthetic_data": synthetic_df.to_dict(orient="records")
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/syntheticTestCaseGenerate")
async def process_csv(file: UploadFile):
    """
    Upload a CSV file, process it, save the validation results to a CSV file,
    and return the results in a JSON response.
    """
    try:
        # Read the uploaded file into a DataFrame
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), dtype=str)

        # Process the DataFrame using GenericCSVTesttestGenerator
        testGenerator.potential_primary_keys = testGenerator.identify_primary_keys(df)
        all_validations = []
        for column in df.columns:
            column_validations = testGenerator.generate_column_validations(df, column)
            all_validations.extend(column_validations)

        # Convert validations to a DataFrame
        results_df = pd.DataFrame(all_validations)

        # Save the results to a CSV file
        output_file = f"results_{file.filename}"
        results_df.to_csv(output_file, index=False)

        # Return the validation results and file path in JSONResponse
        return JSONResponse(content={
            "message": "File processed successfully.",
            "validations": results_df.to_dict(orient="records"),
            "saved_file_path": str(output_file)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
