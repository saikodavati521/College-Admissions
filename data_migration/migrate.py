import os
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
subscription_id = os.getenv("SUBSCRIPTION")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WS_NAME")

train_data_name = "admissions-train-data"
test_data_name = "admissions-test-data"


def main():
    """Main function to register MLTable data assets in Azure ML."""
    # Authenticate using default Azure credentials
    azureSubscription = DefaultAzureCredential()

    # Get a handle to the workspace
    print(f"Connecting to Azure ML workspace: {workspace_name}")
    ml_client = MLClient(
        credential=azureSubscription,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    # Get the absolute path to the model_training directory
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
    print(f"Training data registered: {train_data_asset.name} (version {train_data_asset.version})")

    # Register test data as MLTable data asset
    print("Registering test data as MLTable data asset...")
    test_data_asset = Data(
        path=str(test_data_path),
        type=AssetTypes.MLTABLE,
        description="Test dataset for admissions model (MLTable format)",
        name=test_data_name,
    )

    ml_client.data.create_or_update(test_data_asset)
    print(f"Test data registered: {test_data_asset.name} (version {test_data_asset.version})")

    print("\nData asset registration completed successfully!")


if __name__ == '__main__':
    main()