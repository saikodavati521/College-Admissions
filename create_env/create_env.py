"""Environment Creation Script for College Admissions Model.

This script creates and registers a custom Azure ML environment with
specified dependencies for the College Admissions Model project.

Usage:
    python create_env.py

Environment Variables:
    SUBSCRIPTION_ID: Azure subscription ID
    RESOURCE_GROUP: Azure resource group name
    WS_NAME: Azure ML workspace name
"""
import os
import time

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from config import custom_env_name

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WS_NAME")

# Azure ML base image for environment
BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20250217.v1"

# Conda environment file path
CONDA_FILE_PATH = "create_env/conda.yml"

# Polling interval for build status (seconds)
POLLING_INTERVAL = 30


def main():
    """Create and register custom Azure ML environment.

    This function performs the following steps:
        1. Connects to the Azure ML workspace
        2. Creates an environment with conda dependencies
        3. Registers the environment in Azure ML
        4. Polls the build status until completion

    Raises:
        RuntimeError: If environment build fails or is canceled.
    """
    # Connect to Azure ML workspace
    print(f"Connecting to Azure ML workspace: {WORKSPACE_NAME}")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # Create the environment
    custom_job_env = Environment(
        name=custom_env_name,
        description="Custom environment for College Admissions Model",
        conda_file=CONDA_FILE_PATH,
        image=BASE_IMAGE,
    )

    # Register the environment
    print(f"Registering environment: {custom_env_name}")
    custom_job_env = ml_client.environments.create_or_update(custom_job_env)

    print(
        f"Environment '{custom_job_env.name}' registered to workspace "
        f"(version {custom_job_env.version})"
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
        if hasattr(env_status, "build") and env_status.build:
            build_state = env_status.build.build_state
            print(f"Build status: {build_state}")

            # Check for terminal states
            if build_state in ["Succeeded", "Failed", "Canceled"]:
                if build_state == "Succeeded":
                    print(
                        f"\nEnvironment '{environment_name}' version "
                        f"{environment_version} is ready!"
                    )
                    break
                else:
                    raise RuntimeError(
                        f"Environment build {build_state}. "
                        f"Check Azure ML Studio for details."
                    )
        else:
            # If no build object, environment is likely ready
            print("Environment is ready (no build required)")
            break

        # Wait before polling again
        time.sleep(POLLING_INTERVAL)


if __name__ == "__main__":
    main()