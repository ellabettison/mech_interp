#!/usr/bin/env python3
"""
Script to run the automatic feature finder for refusal nodes.
"""

import logging
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Add the current directory to the path so we can import the feature finder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_feature_finder.feature_finder import FeatureFinder

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Starting automatic feature finder...")
    
    try:
        # Initialize the feature finder
        # You can adjust these parameters:
        # - model_name: the model to analyze
        # - sourceset: the feature source set
        # - n_dataset_to_use: number of prompts to test (default 100)
        feature_finder = FeatureFinder(
            model_name="deepseek-r1-distill-llama-8b",
            sourceset="15-llamascope-slimpj-res-32k",
            n_dataset_to_use=50  # Reduced for faster testing
        )
        
        # Run the feature finding pipeline
        print("Running feature finding pipeline...")
        feature_finder.find_features()
        
        print("Feature finding completed!")
        
    except Exception as e:
        print(f"Error running feature finder: {e}")
        logging.exception("Feature finder failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 