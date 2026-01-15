"""Responsible AI Dashboard Creation Script for College Admissions Model.

This script creates a Responsible AI Insights dashboard for the college admissions
classification model. It constructs a pipeline with explanation, causal analysis,
and error analysis components, then submits it to Azure ML.

Usage:
    python responsible.py

Environment Variables:
    SUBSCRIPTION_ID: Azure subscription ID
    RESOURCE_GROUP: Azure resource group name
    WS_NAME: Azure ML workspace name
"""
import json
import os
import time
import uuid

from azure.ai.ml import Input, MLClient, Output, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import PipelineJob
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from config import (
    compute_cluster,
    custom_environment,
    artifact_path_name,
    rai_experiment_name,
    registered_test_data,
    registered_train_data,
)

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WS_NAME")

# RAI pipeline configuration
TARGET_COLUMN_NAME = "Accept"
CATEGORICAL_FEATURES = json.dumps([
    "Gender",
    "Race_American_Indian_or_Alaska_Native",
    "Race_Asian",
    "Race_Black_or_African_American",
    "Race_Native_Hawaiian_or_Other_Pacific_Islander",
    "Race_White",
    "Ethnicity_Hispanic_or_Latino",
    "LegacyStatus",
    "FirstGeneration"
])
TREATMENT_FEATURES = json.dumps([
    "GPA",
    "SAT",
    "Gender",
    "Race_American_Indian_or_Alaska_Native",
    "Race_Asian",
    "Race_Black_or_African_American",
    "Race_Native_Hawaiian_or_Other_Pacific_Islander",
    "Race_White",
    "Ethnicity_Hispanic_or_Latino",
    "LegacyStatus",
    "FirstGeneration"
])

# Azure ML configuration
RAI_LABEL = "latest"
RAI_REGISTRY_NAME = "azureml"
RAI_REGISTRY_LOCATION = "eastus"

# Timeout configuration
TIMEOUT = 300  # Timeout for each component job in seconds (2 hours)
POLLING_INTERVAL = 30  # Job status polling interval in seconds


def submit_and_wait(ml_client, pipeline_job) -> PipelineJob:
    """Submit pipeline job and wait for completion.

    Args:
        ml_client: Azure ML client instance.
        pipeline_job: Pipeline job to submit.

    Returns:
        PipelineJob: Completed pipeline job.

    Raises:
        AssertionError: If job does not complete successfully.
    """
    created_job = ml_client.jobs.create_or_update(pipeline_job)
    assert created_job is not None

    print("Pipeline job can be accessed in the following URL:")
    print(created_job.studio_url)

    while created_job.status not in [
        "Completed",
        "Failed",
        "Canceled",
        "NotResponding",
    ]:
        time.sleep(POLLING_INTERVAL)
        created_job = ml_client.jobs.get(created_job.name)
        print(f"Latest status: {created_job.status}")

    assert created_job.status == "Completed", (
        f"Pipeline job ended with status: {created_job.status}. "
        f"Check Azure ML Studio for details."
    )
    return created_job


def create_rai_pipeline(
    rai_constructor_component,
    rai_explanation_component,
    rai_causal_component,
    rai_erroranalysis_component,
    rai_gather_component,
    model_id,
    model_path,
    train_data_path,
    test_data_path
):
    """Create RAI classification pipeline with all components.

    Args:
        rai_constructor_component: RAI constructor component.
        rai_explanation_component: RAI explanation component.
        rai_causal_component: RAI causal analysis component.
        rai_erroranalysis_component: RAI error analysis component.
        rai_gather_component: RAI gather component.
        model_id (str): Model identifier string.
        model_path (str): Path to MLflow model.
        train_data_path (str): Path to training MLTable.
        test_data_path (str): Path to test MLTable.

    Returns:
        function: Pipeline function decorated with @dsl.pipeline.
    """
    @dsl.pipeline(
        compute=compute_cluster,
        description=(
            "Responsible AI Insights pipeline for admissions classification model"
        ),
        experiment_name=rai_experiment_name,
    )
    def rai_classification_pipeline():
        # Initiate the RAIInsights
        create_rai_job = rai_constructor_component(
            title="RAI Dashboard - Admissions Model",
            task_type="classification",
            model_info=model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=model_path),
            train_dataset=Input(type=AssetTypes.MLTABLE, path=train_data_path),
            test_dataset=Input(type=AssetTypes.MLTABLE, path=test_data_path),
            target_column_name=TARGET_COLUMN_NAME,
            categorical_column_names=CATEGORICAL_FEATURES,
        )
        create_rai_job.set_limits(timeout=TIMEOUT)

        # Add an explanation
        explain_job = rai_explanation_component(
            comment="Explanation for the classification dataset",
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        explain_job.set_limits(timeout=TIMEOUT)

        # Add causal analysis
        causal_job = rai_causal_component(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
            treatment_features=TREATMENT_FEATURES,
        )
        causal_job.set_limits(timeout=TIMEOUT)

        # Add error analysis
        erroranalysis_job = rai_erroranalysis_component(
            rai_insights_dashboard=create_rai_job.outputs.rai_insights_dashboard,
        )
        erroranalysis_job.set_limits(timeout=TIMEOUT)

        # Combine everything
        rai_gather_job = rai_gather_component(
            constructor=create_rai_job.outputs.rai_insights_dashboard,
            insight_1=explain_job.outputs.explanation,
            insight_2=causal_job.outputs.causal,
            insight_3=erroranalysis_job.outputs.error_analysis,
        )
        rai_gather_job.set_limits(timeout=TIMEOUT)

        return {"ux_json": rai_gather_job.outputs.ux_json}

    return rai_classification_pipeline


def main():
    """Create and submit RAI Insights dashboard.

    This function performs the following steps:
        1. Connects to Azure ML workspace
        2. Retrieves latest model version
        3. Initializes RAI Insights components from Azure ML registry
        4. Creates RAI pipeline with all components
        5. Submits pipeline and waits for completion

    Raises:
        AssertionError: If pipeline job does not complete successfully.
    """
    # Connect to Azure ML workspace
    print(f"Connecting to Azure ML workspace: {WORKSPACE_NAME}")
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # Get handle to Azure ML registry for RAI built-in components
    ml_client_registry = MLClient(
        credential=credential,
        registry_name=RAI_REGISTRY_NAME,
        registry_location=RAI_REGISTRY_LOCATION,
    )

    print("Retrieving latest model version...")

    # Get the latest version of the trained model
    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=artifact_path_name)]
    )

    # Define model paths
    model_id = f"{artifact_path_name}:{latest_model_version}"
    model_path = f"azureml:{artifact_path_name}:{latest_model_version}"

    print(f"Using train data: {registered_train_data}")
    print(f"Using test data: {registered_test_data}")
    print(f"Using model: {model_id}")

    # Initialize RAI Insights components
    print("Initializing RAI Insights components...")
    rai_constructor_component = ml_client_registry.components.get(
        name="rai_tabular_insight_constructor",
        label=RAI_LABEL
    )
    rai_causal_component = ml_client_registry.components.get(
        name="rai_tabular_causal",
        label=RAI_LABEL
    )
    rai_erroranalysis_component = ml_client_registry.components.get(
        name="rai_tabular_erroranalysis",
        label=RAI_LABEL
    )
    rai_explanation_component = ml_client_registry.components.get(
        name="rai_tabular_explanation",
        label=RAI_LABEL
    )
    rai_gather_component = ml_client_registry.components.get(
        name="rai_tabular_insight_gather",
        label=RAI_LABEL
    )
    
    # Create pipeline function
    print("Creating RAI Insights pipeline...")
    rai_pipeline_func = create_rai_pipeline(
        rai_constructor_component=rai_constructor_component,
        rai_explanation_component=rai_explanation_component,
        rai_causal_component=rai_causal_component,
        rai_erroranalysis_component=rai_erroranalysis_component,
        rai_gather_component=rai_gather_component,
        model_id=model_id,
        model_path=model_path,
        train_data_path=registered_train_data,
        test_data_path=registered_test_data
    )

    # Construct the RAI Insights pipeline
    insights_pipeline_job = rai_pipeline_func()

    # Set output path with unique identifier
    unique_code = str(uuid.uuid4())
    insights_pipeline_job.outputs.ux_json = Output(
        path=f"azureml://datastores/workspaceblobstore/paths/ux_json_outputs/{unique_code}/",
        mode="upload",
        type="uri_folder",
    )

    # Submit the pipeline job
    print("Submitting RAI Insights pipeline job...")
    insights_pipeline_job = submit_and_wait(ml_client, insights_pipeline_job)

    print("\n" + "=" * 80)
    print("✓ RAI Insights Dashboard created successfully!")
    print(f"Job name: {insights_pipeline_job.name}")
    print(f"Studio URL: {insights_pipeline_job.studio_url}")
    print("=" * 80)


if __name__ == "__main__":
    main()
