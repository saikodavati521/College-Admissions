"""Model Monitoring Configuration Script for College Admissions Model.

This script configures Azure Monitor alerts for the deployed model endpoint.
It creates an action group for email notifications and sets up metric alerts
for high response times and server exceptions.

Usage:
    python monitor.py

Environment Variables:
    SUBSCRIPTION_ID: Azure subscription ID
    RESOURCE_GROUP: Azure resource group name
    WS_NAME: Azure ML workspace name
    ALERT_EMAIL: Email address for alert notifications
"""
import os
from datetime import timedelta

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.monitor import MonitorManagementClient
from azure.mgmt.monitor.models import (
    ActionGroupResource,
    EmailReceiver,
    MetricAlertAction,
    MetricAlertResource,
    MetricAlertSingleResourceMultipleMetricCriteria,
    MetricCriteria,
)
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Azure ML workspace details from environment variables
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WS_NAME")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")

# Alert configuration constants
ACTION_GROUP_NAME = "Admissions_Alert_Group"
RESPONSE_TIME_ALERT_NAME = "HighServerResponseTime"
SERVER_EXCEPTIONS_ALERT_NAME = "ServerExceptions"

# Alert thresholds
RESPONSE_TIME_THRESHOLD_MS = 180000.0  # 3 minutes in milliseconds
SERVER_EXCEPTION_THRESHOLD = 0  # Alert on any exception

def main():
    """Configure Azure Monitor alerts for the deployed model endpoint.

    This function performs the following steps:
        1. Connects to Azure ML workspace
        2. Retrieves Application Insights resource ID
        3. Creates or updates action group for email notifications
        4. Creates or updates response time metric alert
        5. Creates or updates server exceptions metric alert

    Raises:
        Exception: If alert configuration fails.
    """
    # Connect to Azure ML workspace
    print(f"Connecting to Azure ML workspace: {WORKSPACE_NAME}")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    # Get workspace and Application Insights resource ID
    ws = ml_client.workspaces.get(WORKSPACE_NAME)
    app_insights_resource_id = ws.application_insights
    print(f"✓ Connected to workspace: {WORKSPACE_NAME}")
    print(f"Application Insights ID: {app_insights_resource_id}\n")

    # Initialize Azure Monitor client
    monitor_client = MonitorManagementClient(credential, SUBSCRIPTION_ID)

    # =========================================================================
    # Create or Update Action Group
    # =========================================================================
    print("Configuring action group for alert notifications...")
    action_group = ActionGroupResource(
        location="global",
        group_short_name="Model_Alerts",
        enabled=True,
        email_receivers=[
            EmailReceiver(
                name="EmailEngineer",
                email_address=ALERT_EMAIL,
                use_common_alert_schema=True
            )
        ]
    )

    created_action_group = monitor_client.action_groups.create_or_update(
        resource_group_name=RESOURCE_GROUP,
        action_group_name=ACTION_GROUP_NAME,
        action_group=action_group
    )
    action_group_resource_id = created_action_group.id
    print(f"✓ Action group configured: {ACTION_GROUP_NAME}")
    print(f"  Email notifications will be sent to: {ALERT_EMAIL}\n")

    # =========================================================================
    # Create or Update Response Time Alert
    # =========================================================================
    print("Configuring response time metric alert...")
    response_time_criteria = MetricAlertSingleResourceMultipleMetricCriteria(
        all_of=[
            MetricCriteria(
                name=RESPONSE_TIME_ALERT_NAME,
                metric_name="requests/duration",
                time_aggregation="Maximum",
                operator="GreaterThanOrEqual",
                threshold=RESPONSE_TIME_THRESHOLD_MS,
            )
        ]
    )

    response_time_alert = MetricAlertResource(
        location="global",
        description=(
            "Alert when server response time (requests/duration) exceeds threshold. "
            f"Triggers when maximum response time >= {RESPONSE_TIME_THRESHOLD_MS}ms."
        ),
        severity=2,
        enabled=True,
        scopes=[app_insights_resource_id],
        evaluation_frequency=timedelta(minutes=1),
        window_size=timedelta(minutes=1),
        criteria=response_time_criteria,
        actions=[MetricAlertAction(action_group_id=action_group_resource_id)],
        auto_mitigate=True,
    )

    monitor_client.metric_alerts.create_or_update(
        resource_group_name=RESOURCE_GROUP,
        rule_name=RESPONSE_TIME_ALERT_NAME,
        parameters=response_time_alert,
    )
    print(f"✓ Response time alert configured: {RESPONSE_TIME_ALERT_NAME}")
    print(f"  Threshold: {RESPONSE_TIME_THRESHOLD_MS}ms")
    print(f"  Evaluation: Every 1 minute over 5-minute window\n")

    # =========================================================================
    # Create or Update Server Exceptions Alert
    # =========================================================================
    print("Configuring server exceptions metric alert...")
    server_exceptions_criteria = MetricAlertSingleResourceMultipleMetricCriteria(
        all_of=[
            MetricCriteria(
                name=SERVER_EXCEPTIONS_ALERT_NAME,
                metric_name="exceptions/server",
                time_aggregation="Count",
                operator="GreaterThan",
                threshold=SERVER_EXCEPTION_THRESHOLD,
            )
        ]
    )

    server_exceptions_alert = MetricAlertResource(
        location="global",
        description=(
            "Alert when there is a server exception, indicating an error "
            "occurred during model prediction."
        ),
        severity=4,
        enabled=True,
        scopes=[app_insights_resource_id],
        evaluation_frequency=timedelta(minutes=15),
        window_size=timedelta(minutes=15),
        criteria=server_exceptions_criteria,
        actions=[MetricAlertAction(action_group_id=action_group_resource_id)],
        auto_mitigate=True,
    )

    monitor_client.metric_alerts.create_or_update(
        resource_group_name=RESOURCE_GROUP,
        rule_name=SERVER_EXCEPTIONS_ALERT_NAME,
        parameters=server_exceptions_alert,
    )
    print(f"✓ Server exceptions alert configured: {SERVER_EXCEPTIONS_ALERT_NAME}")
    print(f"  Threshold: > {SERVER_EXCEPTION_THRESHOLD} exceptions")
    print(f"  Evaluation: Every 1 minute over 15-minute window\n")

    print("=" * 80)
    print("Model Monitoring Configuration Complete")
    print("=" * 80)
    print(f"Action Group: {ACTION_GROUP_NAME}")
    print(f"Alert Email: {ALERT_EMAIL}")
    print(f"Alerts Configured:")
    print(f"  1. {RESPONSE_TIME_ALERT_NAME}")
    print(f"  2. {SERVER_EXCEPTIONS_ALERT_NAME}")
    print("=" * 80)


if __name__ == "__main__":
    main()
