import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.monitor.models import (
    MetricAlertResource,
    MetricAlertSingleResourceMultipleMetricCriteria,
    MetricCriteria,
    MetricAlertAction,
)
from datetime import timedelta

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WS_NAME")
action_group_resource_id = os.getenv("ACTION_GROUP_RESOURCE_ID")

credential=DefaultAzureCredential()

response_time_alert_rule_name = "HighServerResponseTime"

server_exceptions_alert_rule_name = "ModelPredictionExceptions"

if __name__ == '__main__':

    ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

    ws = ml_client.workspaces.get(workspace_name)

    # ARM resource id of the Application Insights resource linked to the workspace
    app_insights_resource_id = ws.application_insights

    monitor_client = MonitorManagementClient(credential, subscription_id)

    
    response_time_criteria = MetricAlertSingleResourceMultipleMetricCriteria(
        all_of=[
            MetricCriteria(
                name=response_time_alert_rule_name,
                metric_name="requests/duration",
                time_aggregation="Maximum",   # Avg response time (ms)
                operator="GreaterThanOrEqual",
                threshold=180000.0,               
            )
        ]
    )
    alert = MetricAlertResource(
        location="global",
        description="Alert when server response time (requests/duration) is high.",
        severity=2,
        enabled=True,
        scopes=[app_insights_resource_id],
        evaluation_frequency=timedelta(minutes=1),  # check every 1 minute
        window_size=timedelta(minutes=5),           # evaluate over last 5 minutes
        criteria=response_time_criteria,
        actions=[MetricAlertAction(action_group_id=action_group_resource_id)],
        auto_mitigate=True,
    )
    
    provisioned_alert = monitor_client.metric_alerts.create_or_update(
        resource_group_name=resource_group,
        rule_name=response_time_alert_rule_name,
        parameters=alert,
    )
    print(f"Created/updated metric alert: {response_time_alert_rule_name}")

    server_exceptions_criteria = MetricAlertSingleResourceMultipleMetricCriteria(
        all_of=[
            MetricCriteria(
                name=server_exceptions_alert_rule_name,
                metric_name="exceptions/server",
                time_aggregation="Total",  
                operator="GreaterThan",
                threshold=0,               
            )
        ]
    )
    alert = MetricAlertResource(
        location="global",
        description="Alert when their is a server exception meaning their was an error with the model making it's predictions.",
        severity=4,
        enabled=True,
        scopes=[app_insights_resource_id],
        evaluation_frequency=timedelta(minutes=1),  # check every 1 minute
        window_size=timedelta(minutes=10),           # evaluate over last 10 minutes
        criteria=server_exceptions_criteria,
        actions=[MetricAlertAction(action_group_id=action_group_resource_id)],
        auto_mitigate=True,
    )
    
    provisioned_alert = monitor_client.metric_alerts.create_or_update(
        resource_group_name=resource_group,
        rule_name=server_exceptions_alert_rule_name,
        parameters=alert,
    )
    print(f"Created/updated metric alert: {server_exceptions_alert_rule_name}")
    
