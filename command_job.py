"""Create and submit a command job for College Admissions Model training in Azure ML."""

import os
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import command, Input, MLClient
from azure.identity import DefaultAzureCredential

# Define constants that can be imported by other modules
experiment_name = 'model_training'
artifact_path_name = "admissions_model"

# Import the name of the test and train datasets
from data_migration.migrate import train_data_name, test_data_name

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
subscription_id = os.getenv("SUBSCRIPTION")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WS_NAME")

def main():
    """Main function to create and submit the command job."""
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
    
    latest_data_version = max(
    [int(d.version) for d in ml_client.data.list(name=train_data_name)])

    # Get the latest version of the test data
    latest_test_data_version = max(
        [int(d.version) for d in ml_client.data.list(name=test_data_name)])

    # Define the latest version of the registered model for deployment
    train_data = f"azureml:{train_data_name}:{latest_data_version}"

    # Define the latest version of the registered model for deployment
    test_data = f"azureml:{test_data_name}:{latest_test_data_version}"

    # Define the command job
    job = command(
        inputs=dict(
            train_data=Input(type="mltable", path=train_data),
            test_data=Input(type="mltable", path=test_data),
            n_estimators=1100,
            random_state=42,
            artifact_path_name=artifact_path_name,
        ),
        code="./model_training",  # location of source code
        compute="admissions-compute",
        command="python train.py --train_data ${{inputs.train_data}} --test_data ${{inputs.test_data}} "
                "--n_estimators ${{inputs.n_estimators}} "
                "--random_state ${{inputs.random_state}} "
                "--artifact_path_name ${{inputs.artifact_path_name}}",
        environment="admissions_environment@latest",
        experiment_name=experiment_name,
        display_name="trained_admissions_model" 
    )
    
    # Submit the job
    print("Submitting command job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted. Job name: {returned_job.name}")
    
    # Get a URL for monitoring the job
    studio_url = returned_job.studio_url
    print(f"Monitor your job at {studio_url}")


if __name__ == "__main__":
    main()