"""Responsible AI Dashboard Creation Script.

This script creates a Responsible AI Insights dashboard for the college admissions
classification model. It constructs a pipeline with explanation, causal analysis,
and error analysis components.
"""

import os
import sys
import json
import uuid
import time
from pathlib import Path

from dotenv import load_dotenv
from azure.ai.ml import MLClient, dsl, Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import PipelineJob
from azure.identity import DefaultAzureCredential

# Get path to parent directory
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import the name of the registered model in Azure ML
from command_job import artifact_path_name

# Import the name of the test and train datasets
from data_migration.migrate import train_data_name, test_data_name

# Load environment variables from .env file
load_dotenv()

# Constants
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
TREATMENT_FEATURES = json.dumps(["GPA", "SAT",  "Gender", "Race_American_Indian_or_Alaska_Native", "Race_Asian", "Race_Black_or_African_American", "Race_Native_Hawaiian_or_Other_Pacific_Islander", "Race_White", "Ethnicity_Hispanic_or_Latino", "LegacyStatus", "FirstGeneration"])
RAI_LABEL = "latest"
TIMEOUT = 7200  # Timeout for each component job in seconds (2 hours)
EXPERIMENT_NAME = "Responsible_AI_Insights_Admissions_Model"


def submit_and_wait(ml_client, pipeline_job) -> PipelineJob:
    """
    Submit pipeline job and wait for completion.
    
    Args:
        ml_client: Azure ML client instance
        pipeline_job: Pipeline job to submit
        
    Returns:
        PipelineJob: Completed pipeline job
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
        time.sleep(30)
        created_job = ml_client.jobs.get(created_job.name)
        print(f"Latest status: {created_job.status}")
    
    assert created_job.status == "Completed"
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
    """
    Create RAI classification pipeline with all components.
    
    Args:
        rai_constructor_component: RAI constructor component
        rai_explanation_component: RAI explanation component
        rai_causal_component: RAI causal analysis component
        rai_erroranalysis_component: RAI error analysis component
        rai_gather_component: RAI gather component
        model_id: Model identifier string
        model_path: Path to MLflow model
        train_data_path: Path to training MLTable
        test_data_path: Path to test MLTable
        
    Returns:
        Pipeline function decorated with @dsl.pipeline
    """
    @dsl.pipeline(
        compute="admissions-compute",
        description="Responsible AI Insights pipeline for admissions classification model",
        experiment_name=EXPERIMENT_NAME,
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
    """Main function to create and submit RAI Insights dashboard."""
    # Authenticate to Azure ML
    credential = DefaultAzureCredential()
    
    # Get Azure ML workspace details from environment variables
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace_name = os.getenv("WS_NAME")
    
    print(f"Connecting to Azure ML workspace: {workspace_name}")
    
    # Create ML client
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    # Get handle to azureml registry for the RAI built-in components
    ml_client_registry = MLClient(
        credential=credential,
        registry_name="azureml",
        registry_location="eastus",
    )
    
    print("Retrieving latest data and model versions...")
    
    # Get the latest version of the train data
    latest_data_version = max(
        [int(d.version) for d in ml_client.data.list(name=train_data_name)]
    )
    
    # Get the latest version of the test data
    latest_test_data_version = max(
        [int(d.version) for d in ml_client.data.list(name=test_data_name)]
    )
    
    # Get the latest version of the trained model
    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=artifact_path_name)]
    )
    
    # Define data and model paths
    train_data_path = f"azureml:{train_data_name}:{latest_data_version}"
    test_data_path = f"azureml:{test_data_name}:{latest_test_data_version}"
    model_id = f"{artifact_path_name}:{latest_model_version}"
    model_path = f"azureml:{artifact_path_name}:{latest_model_version}"
    
    print(f"Using train data: {train_data_path}")
    print(f"Using test data: {test_data_path}")
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
        train_data_path=train_data_path,
        test_data_path=test_data_path
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
    
    print("\n" + "="*80)
    print("RAI Insights Dashboard created successfully!")
    print(f"Job name: {insights_pipeline_job.name}")
    print(f"Studio URL: {insights_pipeline_job.studio_url}")
    print("="*80)


if __name__ == "__main__":
    main()



