"""Create and register a custom environment for College Admissions Model in Azure ML."""

import os
import time
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# Change to the script's directory to ensure consistent execution regardless of working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
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

    # Poll the environment build status until completion
    print("\nPolling environment build status...")
    environment_name = custom_job_env.name
    environment_version = custom_job_env.version

    while True:
        # Get the latest environment status
        env_status = ml_client.environments.get(
            name=environment_name,
            version=environment_version
        )

        # Check if environment has a build status
        if hasattr(env_status, 'build') and env_status.build:
            build_state = env_status.build.build_state
            print(f"Latest build status: {build_state}")

            # Check for terminal states
            if build_state in ["Succeeded", "Failed", "Canceled"]:
                if build_state == "Succeeded":
                    print(f"\n✓ Environment '{environment_name}' version "
                          f"{environment_version} is ready!")
                    break
                else:
                    raise RuntimeError(
                        f"Environment build {build_state}. "
                        f"Check Azure ML Studio for details."
                    )
        else:
            # If no build object, environment is likely ready
            print("Environment is ready (no build required).")
            break

        # Wait before polling again
        time.sleep(30)


if __name__ == "__main__":
    main()