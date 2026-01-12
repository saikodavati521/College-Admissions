"""Create and register a custom environment for College Admissions Model in Azure ML."""

import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Change to the script's directory to ensure consistent execution regardless of working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
subscription_id = os.getenv("SUBSCRIPTION")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WS_NAME")

# Set custom environment name
custom_env_name = "admissions_environment"


def main():
    """Main function to create and register the environment."""
    # Authenticate using default Azure credentials
    credential = DefaultAzureCredential()

    # Get a handle to the workspace
    print(f"Connecting to Azure ML workspace: {workspace_name}")
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    # Get the path to the conda.yml file
    conda_file_path = "conda.yml"
    
    # Create the environment
    custom_job_env = Environment(
        name=custom_env_name,
        description="Custom environment for College Admissions Model",
        conda_file=conda_file_path,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20250217.v1",
    )
    
    # Register the environment
    print(f"Registering environment: {custom_env_name}")
    custom_job_env = ml_client.environments.create_or_update(custom_job_env)

    print(
        f"Environment with name {custom_job_env.name} is registered to workspace, "
        f"the environment version is {custom_job_env.version}"
    )


if __name__ == "__main__":
    main()