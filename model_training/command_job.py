"""Command Job Submission Script for College Admissions Model.

This script creates and submits an Azure ML command job to train the
College Admissions Model. It configures the training job with specified
hyperparameters, submits it to Azure ML, and polls until completion.

Usage:
    python command_job.py

Environment Variables:
    SUBSCRIPTION_ID: Azure subscription ID
    RESOURCE_GROUP: Azure resource group name
    WS_NAME: Azure ML workspace name
"""
import os
import time

from azure.ai.ml import Input, MLClient, command
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from config import (

    compute_cluster,
    experiment_name,
    custom_environment,
    artifact_path_name,
    registered_test_data,
    registered_train_data,
    
)

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WS_NAME")


# Training hyperparameters
N_ESTIMATORS = 1100
RANDOM_STATE = 42
N_JOBS = -1
MIN_SAMPLES_SPLIT = 20
MIN_SAMPLES_LEAF = 10
MAX_FEATURES = "sqrt"

# Job polling interval (seconds)
POLLING_INTERVAL = 30


def main():
    """Create and submit Azure ML command job for model training.

    This function performs the following steps:
        1. Connects to the Azure ML workspace
        2. Configures the training job with hyperparameters
        3. Submits the job to Azure ML
        4. Polls the job status until completion
        5. Validates successful completion

    Raises:
        AssertionError: If the job does not complete successfully.
    """
    # Connect to Azure ML workspace
    print(f"Connecting to Azure ML workspace: {WORKSPACE_NAME}")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # Configure the command job
    job = command(
        inputs=dict(
            train_data=Input(type="mltable", path=registered_train_data),
            test_data=Input(type="mltable", path=registered_test_data),
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            artifact_path_name=artifact_path_name,
        ),
        code="model_training/train_code",
        compute=compute_cluster,
        command=(
            "python train.py "
            "--train_data ${{inputs.train_data}} "
            "--test_data ${{inputs.test_data}} "
            "--n_estimators ${{inputs.n_estimators}} "
            "--random_state ${{inputs.random_state}} "
            "--n_jobs ${{inputs.n_jobs}} "
            "--min_samples_split ${{inputs.min_samples_split}} "
            "--min_samples_leaf ${{inputs.min_samples_leaf}} "
            "--max_features ${{inputs.max_features}} "
            "--artifact_path_name ${{inputs.artifact_path_name}}"
        ),
        environment=custom_environment,
        experiment_name=experiment_name,
        display_name="trained_admissions_model"
    )
    
    # Submit the job
    print("Submitting command job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted. Job name: {returned_job.name}")

    # Get monitoring URL
    studio_url = returned_job.studio_url
    print(f"Monitor your job at {studio_url}")

    # Poll job status until completion
    print("\nPolling job status...")
    while returned_job.status not in [
        "Completed",
        "Failed",
        "Canceled",
        "NotResponding",
    ]:
        time.sleep(POLLING_INTERVAL)
        returned_job = ml_client.jobs.get(returned_job.name)
        print(f"Latest status: {returned_job.status}")

    # Validate job completed successfully
    assert returned_job.status == "Completed", (
        f"Job ended with status: {returned_job.status}. "
        f"Check Azure ML Studio for details."
    )
    print("\n✓ Training job completed successfully!")


if __name__ == "__main__":
    main()
