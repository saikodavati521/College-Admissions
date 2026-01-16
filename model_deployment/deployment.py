"""Model Deployment Script for College Admissions Model.

This script deploys the trained college admissions classification model to
an Azure ML managed online endpoint. It configures the deployment with the
latest model version, custom environment, and scoring script.

Usage:
    python deployment.py

Environment Variables:
    SUBSCRIPTION_ID: Azure subscription ID
    RESOURCE_GROUP: Azure resource group name
    WS_NAME: Azure ML workspace name
"""
import json
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import CodeConfiguration, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from config import (
    artifact_path_name,
    custom_env_name,
    deployment_name,
    online_endpoint_name,
)

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Azure ML Workspace Configuration
# =============================================================================
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WS_NAME")

# =============================================================================
# Deployment Configuration
# =============================================================================
INSTANCE_TYPE = "Standard_D2as_v4"
INSTANCE_COUNT = 1

# Required columns for the admissions model
REQUIRED_COLUMNS = [
    "GPA",
    "SAT",
    "Age",
    "Gender",
    "EssayScore",
    "InterviewScore",
    "ExtracurricularScore",
    "RecommendationScore",
    "LegacyStatus",
    "FinancialAid",
    "FirstGeneration",
    "Race_American_Indian_or_Alaska_Native",
    "Race_Asian",
    "Race_Black_or_African_American",
    "Race_Native_Hawaiian_or_Other_Pacific_Islander",
    "Race_White",
    "Ethnicity_Hispanic_or_Latino"
]


def deploy_model():
    """Deploy the college admissions model to Azure ML online endpoint.

    This function performs the following steps:
        1. Connects to Azure ML workspace
        2. Retrieves latest environment version
        3. Retrieves latest model version
        4. Creates managed online deployment with custom environment
        5. Waits for deployment to complete
        6. Sets traffic to 100% for the new deployment

    Raises:
        Exception: If deployment fails.
    """
    # =========================================================================
    # Connect to Azure ML Workspace
    # =========================================================================
    print("=" * 80)
    print("Model Deployment to Azure ML Online Endpoint")
    print("=" * 80)
    print(f"Connecting to Azure ML workspace: {WORKSPACE_NAME}")

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    print(f"✓ Connected to workspace: {WORKSPACE_NAME}\n")

    # =========================================================================
    # Retrieve Latest Environment Version
    # =========================================================================
    print("Retrieving latest environment version...")

    env_versions = [int(e.version) for e in ml_client.environments.list(name=custom_env_name)]
    latest_env_version = max(env_versions)
    latest_env_path = f"azureml:{custom_env_name}:{latest_env_version}"

    print(f"✓ Environment: {latest_env_path}\n")

    # =========================================================================
    # Retrieve Latest Model Version
    # =========================================================================
    print(f"Retrieving latest version of model: {artifact_path_name}")

    model_versions = [int(m.version) for m in ml_client.models.list(name=artifact_path_name)]
    latest_model_version = max(model_versions)
    model_path = f"azureml:{artifact_path_name}:{latest_model_version}"

    print(f"✓ Model: {model_path}\n")

    # =========================================================================
    # Configure Deployment
    # =========================================================================
    print("Configuring deployment...")
    print(f"  Endpoint: {online_endpoint_name}")
    print(f"  Deployment: {deployment_name}")
    print(f"  Instance type: {INSTANCE_TYPE}")
    print(f"  Instance count: {INSTANCE_COUNT}")
    print(f"  App Insights: Enabled\n")

    # Get the absolute path to the scoring_code directory
    scoring_code_path = Path(__file__).parent / "scoring_code"

    admissions_deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=online_endpoint_name,
        model=model_path,
        environment=latest_env_path,
        code_configuration=CodeConfiguration(
            code=str(scoring_code_path),
            scoring_script="score.py",
        ),
        instance_type=INSTANCE_TYPE,
        instance_count=INSTANCE_COUNT,
        environment_variables={
            "MLFLOW_FOLDER_NAME": artifact_path_name,
            "REQUIRED_COLUMNS": json.dumps(REQUIRED_COLUMNS),
        },
        app_insights_enabled=True,
    )

    # =========================================================================
    # Create Deployment
    # =========================================================================
    print(f"Initiating deployment '{deployment_name}'...")
    deployment_result = ml_client.online_deployments.begin_create_or_update(
        admissions_deployment
    )

    print("Waiting for deployment to complete...")
    deployment_result.wait()

    print("\n" + "=" * 80)
    print("✓ Deployment Completed Successfully")
    print("=" * 80)
    print(f"Endpoint: {online_endpoint_name}")
    print(f"Deployment: {deployment_name}")
    print(f"Model: {model_path}")
    print(f"Environment: {latest_env_path}")
    print("=" * 80 + "\n")

    # =========================================================================
    # Configure Traffic
    # =========================================================================
    print("Setting traffic to 100% for the new deployment...")
    endpoint = ml_client.online_endpoints.get(online_endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint)

    print(f"✓ Traffic set to 100% for deployment: {deployment_name}\n")
    print("=" * 80)
    print("✓ Model Deployment Complete")
    print("=" * 80)


if __name__ == "__main__":
    deploy_model()