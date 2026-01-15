"""Config Package for College Admissions Model.

This package provides centralized configuration for the entire project.
All constants, names, and configuration values are defined here.

Usage:
    # Import specific variables
    from config import train_data_name, experiment_name, artifact_path_name

    # Import all variables
    from config import *

    # Import the variables module
    from config import variables
"""

# Import all variables from variables.py to make them available at package level
from .variables import *  # noqa: F401, F403