import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.core.exceptions import ResourceNotFoundError

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WS_NAME")

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# Define the endpoint name
online_endpoint_name = "admissions-endpoint348957"
print(f"Checking for endpoint: {online_endpoint_name}")

# Check if endpoint already exists
try:
    existing_endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)
    print(f"{online_endpoint_name} endpoint is already created")
except ResourceNotFoundError:
    # Endpoint does not exist, create it
    print(f"Endpoint not found. Creating {online_endpoint_name}...")
    
    # Define an online endpoint
    endpoint = ManagedOnlineEndpoint(
        name=online_endpoint_name,
        description="this is an online endpoint for admissions model",
        auth_mode="key",
        tags={
            "training_dataset": "train_admissions_dataset.csv",
            "test_dataset": "test_admissions_dataset.csv",
        },
    )
    
    # Create the online endpoint and poll for completion
    print("Provisioning endpoint...")
    poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    
    # Wait for the operation to complete
    endpoint_result = poller.result()
    
    # Verify endpoint creation
    created_endpoint = ml_client.online_endpoints.get(name=online_endpoint_name)
    print(f"{online_endpoint_name} endpoint is created")