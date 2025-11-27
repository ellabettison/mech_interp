#!/usr/bin/env python3
"""
Script to run the feature finder with relaxed thresholds for finding refusal nodes.
"""

import logging
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to the path so we can import the feature finder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_feature_finder.feature_finder import FeatureFinder

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Starting feature finder with RELAXED thresholds...")
    
    try:
        # Initialize the feature finder
        feature_finder = FeatureFinder(
            model_name="deepseek-r1-distill-llama-8b",
            sourceset="15-llamascope-slimpj-res-32k",
            n_dataset_to_use=100  # Use more prompts
        )
        
        # Override the statistical thresholds to be more lenient
        print("Using relaxed statistical thresholds:")
        print("- P-value threshold: 0.1 (instead of 0.05)")
        print("- Epsilon: 0.1 (instead of 0.2)")
        print("- Min negative mean: 0.3 (instead of 0.5)")
        
        # Run the feature finding pipeline with custom thresholds
        print("Running feature finding pipeline...")
        
        # We need to modify the method call to use relaxed thresholds
        # Let's create a custom analysis function
        results = run_relaxed_analysis(feature_finder)
        
        print(f"Found {len(results)} potential refusal features with relaxed thresholds!")
        
    except Exception as e:
        print(f"Error running feature finder: {e}")
        logging.exception("Feature finder failed")
        return 1
    
    return 0

def run_relaxed_analysis(feature_finder):
    """Run analysis with relaxed thresholds."""
    from collections import defaultdict
    import numpy as np
    from scipy.stats import ttest_ind
    
    print("Running relaxed statistical analysis...")
    
    # Get refused and complied prompts
    refused_prompts = set()
    comply_prompts = set()
    
    for prompt, refused in feature_finder.refusal_cache.items():
        if refused:
            refused_prompts.add(prompt)
        else:
            comply_prompts.add(prompt)
    
    print(f"Refused prompts: {len(refused_prompts)}")
    print(f"Comply prompts: {len(comply_prompts)}")
    
    # Collect activations for all features
    refused_activations = defaultdict(list)
    comply_activations = defaultdict(list)
    
    # Process refused prompts
    for prompt in list(refused_prompts)[:50]:  # Limit to avoid rate limits
        try:
            top_features = feature_finder.get_top_features_for_prompt(prompt)
            for feature in top_features:
                feature_id = feature['featureIndex']
                activation = feature['activationValue']
                refused_activations[feature_id].append(activation)
        except Exception as e:
            print(f"Error processing refused prompt: {e}")
            continue
    
    # Process comply prompts
    for prompt in list(comply_prompts)[:50]:  # Limit to avoid rate limits
        try:
            top_features = feature_finder.get_top_features_for_prompt(prompt)
            for feature in top_features:
                feature_id = feature['featureIndex']
                activation = feature['activationValue']
                comply_activations[feature_id].append(activation)
        except Exception as e:
            print(f"Error processing comply prompt: {e}")
            continue
    
    # Find common features
    common_features = set(refused_activations.keys()) & set(comply_activations.keys())
    print(f"Common features: {len(common_features)}")
    
    # Analyze with relaxed thresholds
    results = []
    relaxed_p_threshold = 0.1  # More lenient
    relaxed_epsilon = 0.1      # More lenient
    relaxed_min_neg = 0.3      # More lenient
    
    for feature_id in common_features:
        refuse_vals = refused_activations[feature_id]
        comply_vals = comply_activations[feature_id]
        
        if len(refuse_vals) < 2 or len(comply_vals) < 2:
            continue
        
        # Statistical test
        stat, p_value = ttest_ind(refuse_vals, comply_vals, equal_var=False)
        refuse_mean = np.mean(refuse_vals)
        comply_mean = np.mean(comply_vals)
        diff = refuse_mean - comply_mean
        
        # Relaxed filtering criteria
        if (comply_mean < relaxed_epsilon and 
            refuse_mean > relaxed_min_neg and 
            p_value < relaxed_p_threshold and
            diff > 0):
            
            results.append({
                "featureIndex": feature_id,
                "diff": diff,
                "p_value": p_value,
                "refuse_mean": refuse_mean,
                "comply_mean": comply_mean,
                "n_refuse": len(refuse_vals),
                "n_comply": len(comply_vals)
            })
    
    # Sort by difference
    results.sort(key=lambda x: x["diff"], reverse=True)
    
    # Save results
    import json
    output_file = f"caches/{feature_finder.model_name}/{feature_finder.sourceset_to_test}/diffs_relaxed.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Print top results
    print("\nTop refusal features (relaxed thresholds):")
    for i, result in enumerate(results[:10], 1):
        print(f"{i}. Feature {result['featureIndex']}: diff={result['diff']:.4f}, p={result['p_value']:.4f}")
    
    return results

if __name__ == "__main__":
    exit(main()) 