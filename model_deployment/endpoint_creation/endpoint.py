"""Azure ML Online Endpoint Creation Script for College Admissions Model.

This script creates a managed online endpoint in Azure ML for deploying
the college admissions classification model.

Usage:
    python endpoint.py

Environment Variables:
    SUBSCRIPTION_ID: Azure subscription ID
    RESOURCE_GROUP: Azure resource group name
    WS_NAME: Azure ML workspace name
"""
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from config import online_endpoint_name

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WS_NAME")


def create_online_endpoint():
    """Create a managed online endpoint in Azure ML.

    This function connects to Azure ML workspace and creates a managed
    online endpoint for model deployment with key-based authentication.

    Returns:
        ManagedOnlineEndpoint: The created endpoint object.

    Raises:
        Exception: If endpoint creation fails.
    """
    print(f"Creating endpoint: {online_endpoint_name}...")

    # Connect to Azure ML workspace
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # Define an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="Online endpoint for college admissions classification model",
        auth_mode="key",
    )

    # Create the online endpoint and poll for completion
    print("Provisioning endpoint...")
    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)

    # Wait for the operation to complete
    endpoint_result = poller.result()

    print(f" {online_endpoint_name} endpoint is created")
    return endpoint_result


if __name__ == "__main__":
    create_online_endpoint()