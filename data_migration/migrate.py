"""Data Migration Script for College Admissions Model.

This script registers training and test datasets as MLTable data assets
in Azure Machine Learning workspace.

Usage:
    python migrate.py

Environment Variables:
    SUBSCRIPTION_ID: Azure subscription ID
    RESOURCE_GROUP: Azure resource group name
    WS_NAME: Azure ML workspace name
"""
import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from config import test_data_name, train_data_name

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WS_NAME")


def main():
    """Register MLTable data assets in Azure ML workspace.

    This function performs the following steps:
        1. Authenticates to Azure using DefaultAzureCredential
        2. Connects to the Azure ML workspace
        3. Registers training data as an MLTable asset
        4. Registers test data as an MLTable asset

    Raises:
        Exception: If authentication or data registration fails.
    """

    # Connect to Azure ML workspace
    print(f"Connecting to Azure ML workspace: {WORKSPACE_NAME}")
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # Get absolute paths to data directories
    parent_dir = Path(__file__).parent.parent
    train_data_path = parent_dir / "model_training" / "train_data"
    test_data_path = parent_dir / "model_training" / "test_data"

    # Register training data as MLTable data asset
    print("Registering training data as MLTable data asset...")
    train_data_asset = Data(
        path=str(train_data_path),
        type=AssetTypes.MLTABLE,
        description="Training dataset for admissions model (MLTable format)",
        name=train_data_name,
    )
    ml_client.data.create_or_update(train_data_asset)
    print(
        f"✓ Training data registered: {train_data_asset.name} "
    )

    # Register test data as MLTable data asset
    print("Registering test data as MLTable data asset...")
    test_data_asset = Data(
        path=str(test_data_path),
        type=AssetTypes.MLTABLE,
        description="Test dataset for admissions model (MLTable format)",
        name=test_data_name,
    )
    ml_client.data.create_or_update(test_data_asset)
    print(
        f"✓ Test data registered: {test_data_asset.name} "
    )

    print("\n✓ Data asset registration completed successfully!")


if __name__ == "__main__":
    main()