"""Model Registration Script for College Admissions Model.

This script finds the latest completed job in the specified experiment and
registers its output as a model in Azure ML using the MLflow runs:/ URI format.
It resolves the MLflow run ID from the Azure ML job and creates a model
registration in the Azure ML Model Registry.

Usage:
    python register.py

Environment Variables:
    SUBSCRIPTION_ID: Azure subscription ID
    RESOURCE_GROUP: Azure resource group name
    WS_NAME: Azure ML workspace name
"""
import os

import mlflow
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from config import artifact_path_name, experiment_name

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WS_NAME")

# MLflow search configuration
MAX_MLFLOW_RUNS = 200


def find_latest_job_in_experiment(ml_client, experiment_name):
    """Find the most recent job in the specified experiment.

    Args:
        ml_client: Azure ML client.
        experiment_name (str): Name of the experiment to search in.

    Returns:
        Job: Latest job object from the specified experiment.

    Raises:
        RuntimeError: If no jobs are found in the experiment.
    """
    filtered_jobs = []

    for job in ml_client.jobs.list():
        if getattr(job, "experiment_name", None) == experiment_name:
            filtered_jobs.append(job)

    if not filtered_jobs:
        raise RuntimeError(f"No jobs found for experiment '{experiment_name}'.")

    filtered_jobs.sort(
        key=lambda job: job.creation_context.created_at,
        reverse=True,
    )

    latest_job = filtered_jobs[0]
    return ml_client.jobs.get(latest_job.name)


def get_mlflow_experiment_id(mlflow_client, experiment_name):
    """Get the MLflow experiment ID for an Azure ML experiment name.

    Args:
        mlflow_client: MLflow tracking client.
        experiment_name (str): Name of the experiment.

    Returns:
        str: MLflow experiment ID.

    Raises:
        RuntimeError: If the experiment is not found.
    """
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None:
        return exp.experiment_id

    experiments = mlflow_client.search_experiments(
        filter_string=f"name = '{experiment_name}'"
    )
    if experiments:
        return experiments[0].experiment_id

    raise RuntimeError(
        f"MLflow experiment '{experiment_name}' was not found in the tracking server. "
        "Double-check the Azure ML job's experiment_name."
    )


def resolve_run_id_for_job(ml_client, latest_job, experiment_name):
    """Resolve the MLflow run ID corresponding to an Azure ML job.

    This function:
        1. Retrieves the MLflow tracking URI from the workspace
        2. Searches MLflow runs in the matching experiment
        3. Matches the Azure job name against MLflow run tag values

    Args:
        ml_client: Azure ML client.
        latest_job: Latest Azure ML job object.
        experiment_name (str): Name of the experiment.

    Returns:
        str: MLflow run ID.

    Raises:
        RuntimeError: If no matching MLflow run is found.
    """
    workspace = ml_client.workspaces.get(ml_client.workspace_name)
    tracking_uri = workspace.mlflow_tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    mlflow_client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = get_mlflow_experiment_id(mlflow_client, experiment_name)

    runs = mlflow_client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=MAX_MLFLOW_RUNS,
    )

    for run in runs:
        tags = run.data.tags or {}
        if latest_job.name in set(tags.values()):
            return run.info.run_id

    raise RuntimeError(
        f"No MLflow runs found for job '{latest_job.name}' in MLflow experiment "
        f"'{experiment_name}'. Ensure your training job logs something with MLflow "
        "(a metric/param/artifact/model) so a run is created and discoverable."
    )


def main():
    """Register the model from the most recent completed job.

    This function performs the following steps:
        1. Validates environment variables
        2. Connects to Azure ML workspace
        3. Finds the latest job in the experiment
        4. Validates job completion status
        5. Resolves the MLflow run ID
        6. Registers the model in Azure ML Model Registry

    Raises:
        RuntimeError: If environment variables are missing, job is not completed,
                     or MLflow run cannot be resolved.
    """
    if not SUBSCRIPTION_ID or not RESOURCE_GROUP or not WORKSPACE_NAME:
        raise RuntimeError(
            "Missing required environment variables. Ensure your .env contains:\n"
            "SUBSCRIPTION_ID=...\n"
            "RESOURCE_GROUP=...\n"
            "WS_NAME=...\n"
        )

    # Connect to Azure ML workspace
    print(f"Connecting to Azure ML workspace: {WORKSPACE_NAME}")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # Find the latest job in the experiment
    print(f"Finding the most recent job in experiment '{experiment_name}'...")
    latest_job = find_latest_job_in_experiment(ml_client, experiment_name)

    # Check job status
    status = str(getattr(latest_job, "status", "unknown")).lower()
    print(f"Found job: {latest_job.name} (status: {status})")

    # Ensure job is completed
    if status not in {"completed"}:
        raise RuntimeError(
            f"Latest job '{latest_job.name}' is not completed (status={status}). "
            "Wait for completion before registering the model."
        )

    # Resolve MLflow run_id for the selected Azure ML job
    run_id = resolve_run_id_for_job(ml_client, latest_job, experiment_name)
    print(f"Resolved MLflow run_id: {run_id}")

    # Construct the model path using the MLflow runs:/ URI format
    model_path = f"runs:/{run_id}/{artifact_path_name}"
    print(f"Registering model '{artifact_path_name}' from:\n  {model_path}")

    # Create the model registration (Azure ML Model Registry)
    model = Model(
        name=artifact_path_name,
        path=model_path,
        type=AssetTypes.MLFLOW_MODEL,
        description=(
            f"Registered from latest job in experiment '{experiment_name}': "
            f"{latest_job.name}"
        ),
        properties={
            "azureml.job_name": latest_job.name,
            "azureml.experiment_name": experiment_name,
            "mlflow.run_id": run_id,
            "artifact_path": artifact_path_name,
        },
    )

    # Register the model
    registered_model = ml_client.models.create_or_update(model)
    print("\n✓ Model registered successfully!")
    print(f"  Name: {registered_model.name}")
    print(f"  Version: {registered_model.version}")
    print(f"  Description: {registered_model.description}")


if __name__ == "__main__":
    main()
