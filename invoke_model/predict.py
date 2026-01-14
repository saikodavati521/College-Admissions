import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Add parent directory to path to import command_job.py
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

#Other Imports 
from model_deployment.deployment import deployment_name
from model_deployment.endpoint_creation.endpoint import online_endpoint_name

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WS_NAME")

if __name__ == '__main__':

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    response = ml_client.online_endpoints.invoke(
        endpoint_name=online_endpoint_name,
        deployment_name=deployment_name,
        request_file="G:\\My Drive\\Admissions Model\\invoke_model\\sample.json",
    )

    print(response)

