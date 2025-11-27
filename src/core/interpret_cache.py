#!/usr/bin/env python3
"""
Script to interpret and display results from the feature finder cache files.
"""

import json
import os
from collections import Counter

def load_cache_file(filepath):
    """Load a cache file and return its contents."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def analyze_refusal_cache(refusal_cache):
    """Analyze the refusal cache to show prompt classification statistics."""
    if not refusal_cache:
        print("❌ No refusal cache found")
        return
    
    total_prompts = len(refusal_cache)
    refused_prompts = sum(1 for refused in refusal_cache.values() if refused)
    complied_prompts = total_prompts - refused_prompts
    
    print(f"📊 PROMPT CLASSIFICATION RESULTS:")
    print(f"   Total prompts tested: {total_prompts}")
    print(f"   Refused prompts: {refused_prompts} ({refused_prompts/total_prompts*100:.1f}%)")
    print(f"   Complied prompts: {complied_prompts} ({complied_prompts/total_prompts*100:.1f}%)")
    
    # Show some example refused prompts
    refused_examples = [prompt for prompt, refused in refusal_cache.items() if refused][:3]
    print(f"\n🔴 Example refused prompts:")
    for i, prompt in enumerate(refused_examples, 1):
        print(f"   {i}. {prompt[:100]}...")

def analyze_top_features_cache(top_features_cache):
    """Analyze the top features cache to show feature frequency."""
    if not top_features_cache:
        print("❌ No top features cache found")
        return
    
    print(f"\n🔍 TOP FEATURES ANALYSIS:")
    print(f"   Prompts with extracted features: {len(top_features_cache)}")
    
    # Count feature frequencies across all prompts
    feature_counter = Counter()
    for prompt_features in top_features_cache.values():
        for feature in prompt_features:
            feature_counter[feature['featureIndex']] += 1
    
    print(f"   Unique features found: {len(feature_counter)}")
    
    # Show most frequent features
    print(f"\n🏆 Most frequent features:")
    for feature_id, count in feature_counter.most_common(10):
        print(f"   Feature {feature_id}: found in {count} prompts")

def analyze_activation_cache(activation_cache):
    """Analyze the feature activation cache."""
    if not activation_cache:
        print("❌ No activation cache found")
        return
    
    print(f"\n⚡ ACTIVATION CACHE ANALYSIS:")
    print(f"   Total activation records: {len(activation_cache)}")
    
    # Show some example activations
    print(f"\n📈 Sample activation values:")
    sample_activations = list(activation_cache.items())[:5]
    for key, activation in sample_activations:
        prompt, feature_id = key.split("||")
        print(f"   {prompt[:50]}... | Feature {feature_id}: {activation.get('activationValue', 'N/A')}")

def analyze_diffs_file(diffs_file):
    """Analyze the final differences file."""
    if not diffs_file:
        print("❌ No diffs file found or file is empty")
        return
    
    print(f"\n🎯 FINAL REFUSAL FEATURES:")
    print(f"   Features with significant differences: {len(diffs_file)}")
    
    if len(diffs_file) == 0:
        print("   ⚠️  No significant refusal features found!")
        print("   This could mean:")
        print("   - The model didn't refuse enough prompts")
        print("   - The features don't show clear refusal patterns")
        print("   - The statistical thresholds were too strict")
        return
    
    # Show top refusal features
    print(f"\n🏆 Top refusal features (by difference):")
    for i, feature in enumerate(diffs_file[:10], 1):
        print(f"   {i}. Feature {feature['featureIndex']}:")
        print(f"      - Difference: {feature['diff']:.4f}")
        print(f"      - P-value: {feature['p_value']:.4e}")
        print(f"      - Refuse mean: {feature['neg_mean']:.4f}")
        print(f"      - Comply mean: {feature['pos_mean']:.4f}")
        if 'explanation' in feature:
            print(f"      - Description: {feature['explanation'][:100]}...")

def main():
    """Main function to analyze all cache files."""
    model_name = "deepseek-r1-distill-llama-8b"
    sourceset = "15-llamascope-slimpj-res-32k"
    cache_dir = f"caches/{model_name}/{sourceset}"
    
    print("🔍 FEATURE FINDER CACHE INTERPRETATION")
    print("=" * 50)
    
    # Load all cache files
    refusal_cache = load_cache_file(f"{cache_dir}/refusal_cache.json")
    top_features_cache = load_cache_file(f"{cache_dir}/top_features_cache.json")
    activation_cache = load_cache_file(f"{cache_dir}/feature_activations_cache.json")
    diffs_file = load_cache_file(f"{cache_dir}/diffs.json")
    
    # Analyze each cache file
    analyze_refusal_cache(refusal_cache)
    analyze_top_features_cache(top_features_cache)
    analyze_activation_cache(activation_cache)
    analyze_diffs_file(diffs_file)
    
    print("\n" + "=" * 50)
    print("💡 INTERPRETATION GUIDE:")
    print("   • Refused prompts: Prompts that triggered safety mechanisms")
    print("   • Feature activations: How strongly each neuron/feature fired")
    print("   • Differences: How much more a feature activates on refusals vs compliance")
    print("   • P-values: Statistical significance (lower = more significant)")
    print("   • Higher differences = stronger refusal signals")

if __name__ == "__main__":
    main() 