"""Model Prediction Script for College Admissions Model.

This script invokes the deployed Azure ML online endpoint to get predictions
from the college admissions classification model using sample input data.

Usage:
    python predict.py

Environment Variables:
    SUBSCRIPTION_ID: Azure subscription ID
    RESOURCE_GROUP: Azure resource group name
    WS_NAME: Azure ML workspace name
"""
import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from config import deployment_name, online_endpoint_name

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WS_NAME")

# Request file path
REQUEST_FILE = "invoke_model/sample.json"

def invoke_endpoint():
    """Invoke the deployed model endpoint with sample data.

    This function connects to Azure ML workspace, invokes the online endpoint
    with sample input data from a JSON file, and prints the prediction response.

    Returns:
        str: JSON response containing predictions from the model.

    Raises:
        Exception: If endpoint invocation fails.
    """
    # Connect to Azure ML workspace
    print(f"Connecting to Azure ML workspace: {WORKSPACE_NAME}")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # Invoke the online endpoint with sample data
    print(f"Invoking endpoint: {online_endpoint_name}")
    print(f"Deployment: {deployment_name}")
    print(f"Request file: {REQUEST_FILE}")

    response = ml_client.online_endpoints.invoke(
        endpoint_name=online_endpoint_name,
        deployment_name=deployment_name,
        request_file=REQUEST_FILE,
    )

    print("\n✓ Prediction Response:")
    print(response)

    return response


if __name__ == "__main__":
    invoke_endpoint()
