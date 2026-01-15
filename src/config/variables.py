"""Central Configuration File for College Admissions Model Project.

This file contains all shared constants, names, and configuration values
used across the project. Import from this file instead of duplicating values.

Usage:
    from config import train_data_name, experiment_name, artifact_path_name
"""

# =============================================================================
# Data Migration
# =============================================================================
train_data_name = "admissions-train-data"
test_data_name = "admissions-test-data"

# Registered data assets (latest versions)
registered_train_data = f"azureml:{train_data_name}:latest"
registered_test_data = f"azureml:{test_data_name}:latest"


# =============================================================================
# Environment Configuration
# =============================================================================
custom_env_name = "admissions_environment"
custom_environment = f"azureml:{custom_env_name}@latest"


# =============================================================================
# Model Training Configuration
# =============================================================================
experiment_name = "model_training"
artifact_path_name = "admissions_model"


# =============================================================================
# Compute Configuration
# =============================================================================
compute_cluster = "admissions-compute"


# =============================================================================
# Deployment Configuration
# =============================================================================
online_endpoint_name = "admissions-endpoint348957"
deployment_name = "admissions-deployment"


# =============================================================================
# Responsible AI Dashboard Configuration
# =============================================================================
rai_experiment_name = "Responsible_AI_Insights_Admissions_Model"

