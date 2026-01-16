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
    artifact_path_name,
    compute_cluster,
    custom_environment,
    experiment_name,
    test_data_name,
    train_data_name,
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
# Training Hyperparameters
# =============================================================================
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
        2. Retrieves latest data asset versions
        3. Configures the training job with hyperparameters
        4. Submits the job to Azure ML
        5. Polls the job status until completion
        6. Validates successful completion

    Raises:
        AssertionError: If the job does not complete successfully.
    """
    # =========================================================================
    # Connect to Azure ML Workspace
    # =========================================================================
    print("=" * 80)
    print("Azure ML Command Job Submission")
    print("=" * 80)
    print(f"Connecting to Azure ML workspace: {WORKSPACE_NAME}")
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
    print(f"✓ Connected to workspace: {WORKSPACE_NAME}\n")

    # =========================================================================
    # Retrieve Latest Data Asset Versions
    # =========================================================================
    print("Retrieving latest data asset versions...")
    
    # Get the latest version of the train data
    train_versions = [int(d.version) for d in ml_client.data.list(name=train_data_name)]
    latest_train_data_version = max(train_versions)
    train_data_path = f"azureml:{train_data_name}:{latest_train_data_version}"
    
    # Get the latest version of the test data
    test_versions = [int(d.version) for d in ml_client.data.list(name=test_data_name)]
    latest_test_data_version = max(test_versions)
    test_data_path = f"azureml:{test_data_name}:{latest_test_data_version}"
    
    print(f"✓ Train data: {train_data_path}")
    print(f"✓ Test data: {test_data_path}\n")

    # =========================================================================
    # Configure Command Job
    # =========================================================================
    print("Configuring training job...")
    print(f"  Experiment: {experiment_name}")
    print(f"  Compute: {compute_cluster}")
    print(f"  Environment: {custom_environment}")
    print(f"  Hyperparameters:")
    print(f"    - n_estimators: {N_ESTIMATORS}")
    print(f"    - random_state: {RANDOM_STATE}")
    print(f"    - n_jobs: {N_JOBS}")
    print(f"    - min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"    - min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"    - max_features: {MAX_FEATURES}\n")
    
    job = command(
        inputs=dict(
            train_data=Input(type="mltable", path=train_data_path),
            test_data=Input(type="mltable", path=test_data_path),
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
        display_name="trained_admissions_model",
    )
    
    # =========================================================================
    # Submit Job
    # =========================================================================
    print("Submitting command job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"✓ Job submitted successfully")
    print(f"  Job name: {returned_job.name}")
    print(f"  Studio URL: {returned_job.studio_url}\n")

    # =========================================================================
    # Poll Job Status
    # =========================================================================
    print("Polling job status...")
    print(f"Checking every {POLLING_INTERVAL} seconds\n")
    
    terminal_statuses = ["Completed", "Failed", "Canceled", "NotResponding"]
    
    while returned_job.status not in terminal_statuses:
        time.sleep(POLLING_INTERVAL)
        returned_job = ml_client.jobs.get(returned_job.name)
        print(f"  Status: {returned_job.status}")

    # =========================================================================
    # Validate Completion
    # =========================================================================
    print("\n" + "=" * 80)
    if returned_job.status == "Completed":
        print("✓ Training Job Completed Successfully")
        print("=" * 80)
        print(f"Job Name: {returned_job.name}")
        print(f"Experiment: {experiment_name}")
        print(f"Final Status: {returned_job.status}")
        print("=" * 80)
    else:
        print("❌ Training Job Failed")
        print("=" * 80)
        print(f"Job Name: {returned_job.name}")
        print(f"Final Status: {returned_job.status}")
        print(f"Studio URL: {returned_job.studio_url}")
        print("=" * 80)
        raise AssertionError(
            f"Job ended with status: {returned_job.status}. "
            f"Check Azure ML Studio for details."
        )


if __name__ == "__main__":
    main()
