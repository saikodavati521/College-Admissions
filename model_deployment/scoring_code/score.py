import logging
import os
import sys
import json
import uuid
import pandas as pd
import mlflow

def init():
    """
    This function is called when the container is initialized/started,
    typically after create/update of the deployment.
    Loads the registered MLflow model into memory for inference.
    """
    global model
    global logger
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Add StreamHandler for Azure ML container logs
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    
    # Create production-ready formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(stream_handler)
    
    logger.info("Logger initialized successfully")
    logger.info("Initializing admissions model deployment...")
    
    # AZUREML_MODEL_DIR is an environment variable created during deployment
    # It points to the path where the registered model is stored
    # Azure ML places the model at: /var/azureml-app/azureml-models/{model_name}/{version}/
    # MLFLOW_FOLDER_NAME is a custom environment variable set in deployment.py
    mlflow_folder_name = os.getenv("MLFLOW_FOLDER_NAME")
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), mlflow_folder_name)
    
    logger.info("Loading model from path: %s", model_path)
    
    # Load the MLflow model (pipeline with preprocessing + classifier)
    model = mlflow.pyfunc.load_model(model_path)
    
    logger.info("Model loaded successfully - Ready for inference")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform
    the actual scoring/prediction.
    Extracts data from JSON request using Azure ML standard format,
    converts to DataFrame, and returns predictions.
    
    Expected JSON format (Azure ML standard for MLflow models):
    {
        "input_data": {
            "columns": [
                "GPA", "SAT", "Age", "Gender_Male", "Gender_Female",
                "Race_Asian", "Race_Black", "Race_Hispanic", "Race_White", "Race_Other",
                "ExtracurricularScore", "EssayScore", "RecommendationScore",
                "InterviewScore", "LegacyStatus", "FirstGeneration", "FinancialAid"
            ],
            "data": [
                [3.57, 1443, 18, 1, 0, 0, 0, 0, 1, 0, 5, 3, 7, 3, 0, 0, 1],
                [3.03, 1038, 17, 1, 0, 0, 0, 0, 1, 0, 5, 8, 6, 6, 0, 0, 3]
            ],
            "index": [0, 1]
        }
    }
    """
    # Generate unique request ID for traceability
    request_id = str(uuid.uuid4())
    
    logger.info("[RequestID: %s] Request received for admissions prediction", request_id)
    
    try:
        # Parse the JSON request
        json_data = json.loads(raw_data)
        
        # Log input data received
        num_records = len(json_data.get("input_data", {}).get("data", []))
        logger.info(
            "[RequestID: %s] Input data received - Number of records: %d",
            request_id,
            num_records
        )
        
        # Extract input_data from the request
        if "input_data" not in json_data:
            raise ValueError("Request must contain an 'input_data' key")
        
        input_data = json_data["input_data"]
        
        # Validate required keys in input_data
        if "columns" not in input_data:
            raise ValueError("input_data must contain a 'columns' key")
        if "data" not in input_data:
            raise ValueError("input_data must contain a 'data' key")
        
        columns = input_data["columns"]
        data = input_data["data"]
        index = input_data.get("index", None)
        
        # Convert to pandas DataFrame using columns and data
        input_df = pd.DataFrame(data, columns=columns, index=index)
        
        # Define required columns for the admissions model
        required_columns = [
            'GPA', 'SAT', 'Age', 'Gender',
            'EssayScore', 'InterviewScore', 'ExtracurricularScore', 'RecommendationScore',
            'LegacyStatus', 'FinancialAid', 'FirstGeneration',
            'Race_American_Indian_or_Alaska_Native', 'Race_Asian', 'Race_Black_or_African_American',
            'Race_Native_Hawaiian_or_Other_Pacific_Islander', 'Race_White', 'Ethnicity_Hispanic_or_Latino'
        ]
        
        # Validate that all required columns are present
        missing_columns = set(required_columns) - set(input_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure columns are in the correct order
        input_df = input_df[required_columns]
        
        # Make predictions using the loaded model
        predictions = model.predict(input_df)
        
        # Log predictions
        logger.info(
            "[RequestID: %s] Predictions generated - Results: %s",
            request_id,
            predictions.tolist()
        )
        
        # Format the response
        result = {
            "predictions": predictions.tolist(),
            "request_id": request_id
        }
        
        logger.info(
            "[RequestID: %s] Request processed successfully",
            request_id
        )
        return json.dumps(result)
        
    except ValueError as e:
        # Handle validation errors (bad request) - return 400 status
        logger.warning(
            "[RequestID: %s] Validation error (Bad Request): %s",
            request_id,
            str(e)
        )
        
        error_response = {
            "error": str(e),
            "error_type": "ValidationError",
            "request_id": request_id
        }
        return json.dumps(error_response), 400
        
    except Exception as e:
        # Handle unexpected server errors - return 500 status
        logger.exception(
            "[RequestID: %s] Unexpected server error: %s",
            request_id,
            str(e)
        )
        
        error_response = {
            "error": str(e),
            "error_type": "ServerError",
            "request_id": request_id
        }
        return json.dumps(error_response), 500