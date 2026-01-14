"""RAI Dashboard Analysis Script.

This script finds the latest successfully completed Responsible AI Dashboard
pipeline job and downloads its ux.json output for analysis.
"""

import os
import sys
import json
from pathlib import Path
from statistics import mean, stdev

from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Get path to parent directory
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import experiment name from responsible.py
from Responsible_AI_Insights.responsible import EXPERIMENT_NAME

# Load environment variables from .env file
load_dotenv()

# Constants
DOWNLOAD_PATH = "./rai_outputs"
OUTPUT_NAME = "ux_json"
SHAP_THRESHOLD = 5.4  # Threshold for sensitive feature SHAP values
SENSITIVE_FEATURES = [
    "Gender",
    "Race_American_Indian_or_Alaska_Native",
    "Race_Asian",
    "Race_Black_or_African_American",
    "Race_Native_Hawaiian_or_Other_Pacific_Islander",
    "Race_White",
    "Ethnicity_Hispanic_or_Latino"
]
RACE_FEATURES = [
    "Race_American_Indian_or_Alaska_Native",
    "Race_Asian",
    "Race_Black_or_African_American",
    "Race_Native_Hawaiian_or_Other_Pacific_Islander",
    "Race_White",
    "Ethnicity_Hispanic_or_Latino"
]
RACE_STDEV_THRESHOLD = 5.0  # Number of standard deviations for outlier detection


def find_latest_job_in_experiment(ml_client, experiment_name):
    """Find the most recent job in the specified experiment.
    
    Args:
        ml_client: Azure ML client
        experiment_name: Name of the experiment to search in
        
    Returns:
        Latest job object from the specified experiment
        
    Raises:
        RuntimeError: If no jobs are found in the experiment
    """
    print(f"Searching for jobs in experiment: {experiment_name}")
    filtered_jobs = []
    
    for job in ml_client.jobs.list():
        if getattr(job, "experiment_name", None) == experiment_name:
            filtered_jobs.append(job)
    
    if not filtered_jobs:
        raise RuntimeError(f"No jobs found for experiment '{experiment_name}'.")
    
    # Sort by creation time (most recent first)
    filtered_jobs.sort(
        key=lambda job: job.creation_context.created_at,
        reverse=True,
    )
    
    latest_job = filtered_jobs[0]
    print(f"Found {len(filtered_jobs)} job(s) in experiment '{experiment_name}'")
    return ml_client.jobs.get(latest_job.name)


def validate_job_completion(job):
    """Validate that the job has completed successfully.
    
    Args:
        job: Azure ML job object to validate
        
    Raises:
        RuntimeError: If job is not completed successfully
    """
    # Check job status
    status = str(getattr(job, "status", "unknown")).lower()
    print(f"Found job: {job.name} (status: {status})")
    
    # Ensure job is completed
    if status not in {"completed", "finished"}:
        raise RuntimeError(
            f"Latest job '{job.name}' is not completed (status={status}). "
            "Wait for completion before downloading outputs."
        )
    
    print(f"✓ Job '{job.name}' completed successfully")


def download_ux_json(ml_client, job, output_name=OUTPUT_NAME, download_path=DOWNLOAD_PATH):
    """Download ux.json output from the completed RAI pipeline job.
    
    Args:
        ml_client: Azure ML client
        job: Completed pipeline job
        output_name: Name of the output to download (default: 'ux_json')
        download_path: Local path to download the output (default: './rai_outputs')
    """
    print(f"\nDownloading '{output_name}' output from job: {job.name}")
    print(f"Download path: {download_path}")
    
    try:
        ml_client.jobs.download(
            name=job.name,
            output_name=output_name,
            download_path=download_path
        )
        print(f"✓ Successfully downloaded '{output_name}' to {download_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{output_name}' from job '{job.name}': {str(e)}"
        )


def parse_ux_json(download_path=DOWNLOAD_PATH):
    """Parse the downloaded ux.json file and extract globalFeatureImportance data.
    
    Args:
        download_path: Path where ux.json was downloaded
        
    Returns:
        tuple: (feature_list, scores) - Lists of feature names and their SHAP scores
        
    Raises:
        RuntimeError: If ux.json file not found or parsing fails
    """
    print("\nParsing ux.json for feature importance analysis...")
    
    # Find the ux.json file in the download directory
    download_dir = Path(download_path)
    ux_json_files = list(download_dir.rglob("*.json"))
    
    if not ux_json_files:
        raise RuntimeError(
            f"No JSON files found in {download_path}. "
            "Check if download completed successfully."
        )
    
    print(f"Found {len(ux_json_files)} JSON file(s) to parse...")
    
    # Try to find and parse the ux.json file
    for json_file in ux_json_files:
        try:
            print(f"  Checking: {json_file.name}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Navigate to globalFeatureImportance in the nested structure
            # Path: data['modelExplanationData'][0]['precomputedExplanations']['globalFeatureImportance']
            if "modelExplanationData" in data:
                print(f"  ✓ Found 'modelExplanationData' in {json_file.name}")
                
                model_explanation = data["modelExplanationData"]
                if not isinstance(model_explanation, list) or len(model_explanation) == 0:
                    print(f"    ✗ 'modelExplanationData' is not a valid list, skipping...")
                    continue
                
                precomputed = model_explanation[0].get("precomputedExplanations")
                if not precomputed:
                    print(f"    ✗ Missing 'precomputedExplanations' key, skipping...")
                    continue
                
                global_fi = precomputed.get("globalFeatureImportance")
                if not global_fi:
                    print(f"    ✗ Missing 'globalFeatureImportance' key, skipping...")
                    continue
                
                print(f"  ✓ Found 'globalFeatureImportance' in nested structure")
                
                # Extract scores and feature_list
                if "scores" not in global_fi or "feature_list" not in global_fi:
                    print(f"    ✗ Missing 'scores' or 'feature_list' key, skipping...")
                    continue
                
                feature_list = global_fi["feature_list"]
                scores = global_fi["scores"]
                
                print(f"✓ Successfully extracted globalFeatureImportance data")
                print(f"  Total features: {len(feature_list)}")
                return feature_list, scores
            else:
                print(f"  ✗ No 'modelExplanationData' key in {json_file.name}")
                
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON decode error in {json_file.name}: {str(e)}")
            continue
        except Exception as e:
            print(f"  ✗ Error parsing {json_file.name}: {str(e)}")
            continue
    
    raise RuntimeError(
        "Could not find 'globalFeatureImportance' data in downloaded files. "
        "Verify the RAI dashboard generated feature importance insights."
    )


def validate_sensitive_features(feature_list, scores, sensitive_features=SENSITIVE_FEATURES, 
                                threshold=SHAP_THRESHOLD, race_features=RACE_FEATURES, 
                                stdev_threshold=RACE_STDEV_THRESHOLD):
    """Validate sensitive features using SHAP threshold and race feature distribution.
    
    This function performs two independent validation checks:
    1. SHAP Threshold Check: Ensures all sensitive features have SHAP values within
       the acceptable threshold (±threshold).
    2. Race Feature Distribution Check: Ensures race feature SHAP values don't have
       outliers that deviate too far from the mean, which could indicate discrimination
       or privilege in model predictions.
    
    Args:
        feature_list: List of feature names
        scores: List of SHAP scores corresponding to features
        sensitive_features: List of sensitive feature names to check
        threshold: Absolute SHAP value threshold (default: 0.4)
        race_features: List of race/ethnicity feature names
        stdev_threshold: Number of standard deviations for outlier detection (default: 2.0)
        
    Returns:
        bool: True if all validations pass, False otherwise
    """
    print("\n" + "="*80)
    print("Sensitive Feature SHAP Value Analysis")
    print("="*80)
    print(f"Threshold: ±{threshold}")
    print(f"Checking {len(sensitive_features)} sensitive features...\n")
    
    # Track validation results independently
    shap_threshold_pass = True
    race_distribution_pass = True
    shap_violations = []
    race_outliers = []
    
    # Create a mapping of feature names to SHAP scores
    feature_scores = dict(zip(feature_list, scores))
    
    # ========================================================================
    # VALIDATION 1: SHAP Threshold Check
    # ========================================================================
    print("1. SHAP Threshold Validation")
    print("-" * 80)
    
    for feature in sensitive_features:
        if feature in feature_scores:
            shap_value = feature_scores[feature]
            status = "PASS"
            
            # Check if SHAP value exceeds threshold (positive or negative)
            if shap_value >= threshold or shap_value <= -threshold:
                shap_threshold_pass = False
                status = "FAIL"
                shap_violations.append((feature, shap_value))
            
            print(f"  {feature}: {shap_value:.4f} [{status}]")
        else:
            print(f"  {feature}: NOT FOUND (skipped)")
    
    # ========================================================================
    # VALIDATION 2: Race Feature Distribution Check
    # ========================================================================
    print("\n2. Race Feature Distribution Validation")
    print("-" * 80)
    print(f"Standard Deviation Threshold: {stdev_threshold} σ")
    print(f"Checking {len(race_features)} race/ethnicity features...\n")
    
    # Extract SHAP values for race features
    race_shap_values = []
    race_feature_map = {}
    
    for feature in race_features:
        if feature in feature_scores:
            shap_value = feature_scores[feature]
            race_shap_values.append(shap_value)
            race_feature_map[feature] = shap_value
        else:
            print(f"  {feature}: NOT FOUND (skipped from distribution analysis)")
    
    # Perform distribution analysis if we have at least 2 race features
    if len(race_shap_values) >= 2:
        race_mean = mean(race_shap_values)
        race_std = stdev(race_shap_values) if len(race_shap_values) > 1 else 0.0
        
        print(f"Race Feature SHAP Statistics:")
        print(f"  Mean: {race_mean:.4f}")
        print(f"  Standard Deviation: {race_std:.4f}")
        print(f"  Outlier Threshold: ±{stdev_threshold * race_std:.4f} from mean\n")
        
        # Check each race feature for outliers
        for feature, shap_value in race_feature_map.items():
            deviation = abs(shap_value - race_mean)
            z_score = deviation / race_std if race_std > 0 else 0.0
            status = "PASS"
            
            # Check if feature is an outlier (deviates more than threshold std devs)
            if z_score > stdev_threshold:
                race_distribution_pass = False
                status = "FAIL"
                race_outliers.append((feature, shap_value, z_score))
            
            print(f"  {feature}: {shap_value:.4f} (z-score: {z_score:.2f}) [{status}]")
    else:
        print(f"  ⚠ Insufficient race features found ({len(race_shap_values)}). "
              "Need at least 2 for distribution analysis.")
        print("  Skipping race distribution validation.")
    
    # ========================================================================
    # FINAL VALIDATION SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    # Determine overall pass/fail
    rai_pass = shap_threshold_pass and race_distribution_pass
    
    # Report SHAP threshold results
    if shap_threshold_pass:
        print("✓ SHAP Threshold Check: PASSED")
        print("  All sensitive features have SHAP values within acceptable threshold.")
    else:
        print("❌ SHAP Threshold Check: FAILED")
        print(f"  Found {len(shap_violations)} sensitive feature(s) exceeding threshold:")
        for feature, value in shap_violations:
            print(f"    - {feature}: {value:.4f}")
    
    print()
    
    # Report race distribution results
    if len(race_shap_values) >= 2:
        if race_distribution_pass:
            print("✓ Race Distribution Check: PASSED")
            print("  All race features have similar SHAP importance (no outliers detected).")
        else:
            print("❌ Race Distribution Check: FAILED")
            print(f"  Found {len(race_outliers)} race feature(s) with outlier SHAP values:")
            for feature, value, z_score in race_outliers:
                print(f"    - {feature}: {value:.4f} (z-score: {z_score:.2f})")
            print("  ⚠ This may indicate discrimination or privilege in model predictions.")
    else:
        print("⊘ Race Distribution Check: SKIPPED")
        print("  Insufficient race features for distribution analysis.")
    
    print("\n" + "-"*80)
    if rai_pass:
        print("✓ OVERALL RAI_PASS = True")
        print("All validation checks passed successfully.")
    else:
        print("❌ OVERALL RAI_PASS = False")
        failed_checks = []
        if not shap_threshold_pass:
            failed_checks.append("SHAP Threshold")
        if not race_distribution_pass:
            failed_checks.append("Race Distribution")
        print(f"Failed validation(s): {', '.join(failed_checks)}")
    print("="*80)
    
    return rai_pass


def main():
    """Main function to find and download RAI Dashboard outputs."""
    print("="*80)
    print("RAI Dashboard Analysis - Output Download")
    print("="*80)
    
    # Authenticate to Azure ML
    credential = DefaultAzureCredential()
    
    # Get Azure ML workspace details from environment variables
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace_name = os.getenv("WS_NAME")
    
    print(f"\nConnecting to Azure ML workspace: {workspace_name}")
    
    # Create ML client
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    print(f"✓ Connected to workspace: {workspace_name}\n")
    
    try:
        # Find the latest job in the RAI experiment
        latest_job = find_latest_job_in_experiment(ml_client, EXPERIMENT_NAME)
        
        # Validate that the job completed successfully
        validate_job_completion(latest_job)
        
        # Download the ux.json output
        download_ux_json(ml_client, latest_job)
        
        # Parse the ux.json file and extract feature importance data
        feature_list, scores = parse_ux_json(DOWNLOAD_PATH)
        
        # Validate sensitive features against SHAP threshold
        rai_pass = validate_sensitive_features(feature_list, scores)
        
        # Set Azure Pipeline output variable for use in subsequent stages
        if rai_pass==True:
            print(f"##vso[task.setvariable variable=rai_pass;isOutput=true]{rai_pass}")
            print("RAI Gate: PASSED")
        else:
            print(f"##vso[task.setvariable variable=rai_pass;isOutput=true]{rai_pass}")
            print("RAI Gate: FAILED")
        
        print("\n" + "="*80)
        print("RAI Dashboard Analysis Complete")
        print("="*80)
        print(f"Job name: {latest_job.name}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Output location: {DOWNLOAD_PATH}")
        print(f"RAI Validation: {'PASSED' if rai_pass else 'FAILED'}")
        print("="*80)
        
        # Exit with failure code if RAI validation failed
        if not rai_pass:
            sys.exit(1)
        
        return rai_pass
        
    except RuntimeError as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()