import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineDeployment,  CodeConfiguration

# Add parent directory to path to import command_job.py
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Now import from command_job after adding the parent directory to sys.path
from command_job import artifact_path_name
from create_env.create_env import custom_env_name
from model_deployment.endpoint_creation.endpoint import online_endpoint_name

# Load environment variables from .env file
load_dotenv()

# Define deployment name
deployment_name = "admissions-deployment"

# Define required columns for the admissions model
required_columns = [
    'GPA', 'SAT', 'Age', 'Gender',
    'EssayScore', 'InterviewScore', 'ExtracurricularScore', 
    'RecommendationScore',
    'LegacyStatus', 'FinancialAid', 'FirstGeneration',
    'Race_American_Indian_or_Alaska_Native', 'Race_Asian', 
    'Race_Black_or_African_American',
    'Race_Native_Hawaiian_or_Other_Pacific_Islander', 'Race_White', 
    'Ethnicity_Hispanic_or_Latino'
]

# Get Azure ML workspace details from environment variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WS_NAME")

if __name__ == '__main__':
    # Create ML client
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    # Get the latest version of the trained model
    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=artifact_path_name)])

    # Define the latest version of the registered model for deployment
    model = f"azureml:{artifact_path_name}:{latest_model_version}"

    # Get the latest version of the environment
    latest_env = max(
        ml_client.environments.list(name=custom_env_name),
        key=lambda e: int(e.version),
    )
    environment = f"azureml:{latest_env.name}:{latest_env.version}"

    # Get the absolute path to the scoring_code directory
    scoring_code_path = Path(__file__).parent / "scoring_code"

    # Define an online deployment
    admissions_deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=online_endpoint_name,
        model=model,
        environment=environment,
        code_configuration=CodeConfiguration(
            code=str(scoring_code_path),
            scoring_script="score.py"
        ),
        instance_type="Standard_D2as_v4",
        instance_count=1,
        environment_variables={
            "MLFLOW_FOLDER_NAME": artifact_path_name,
            "REQUIRED_COLUMNS": json.dumps(required_columns),
        },
        app_insights_enabled=True,
    )

    # Create the online deployment
    deployment_result = ml_client.online_deployments.begin_create_or_update(
        admissions_deployment
    )

    print(f"Deployment '{admissions_deployment.name}' initiated...")
    print("Waiting for deployment to complete...")
    deployment_result.wait()
    print(f"Deployment completed successfully!")
    print(f"Endpoint: {online_endpoint_name}")
    print(f"Deployment: {admissions_deployment.name}")

    # Set traffic to 100% for the new deployment
    print("Setting traffic to 100% for the new deployment...")
    endpoint = ml_client.online_endpoints.get(online_endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint)
    print(f"Traffic set to 100% for deployment: {deployment_name}")