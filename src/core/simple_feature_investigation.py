#!/usr/bin/env python3
"""
Simple script to investigate Feature 18677 using cache data.
"""

import json
import os
from collections import defaultdict

def load_cache_file(filepath):
    """Load a cache file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def investigate_feature_18677():
    """Investigate Feature 18677 using our cache data."""
    
    print("🔍 INVESTIGATING FEATURE 18677")
    print("=" * 50)
    
    # Load cache data
    model_name = "deepseek-r1-distill-llama-8b"
    sourceset = "15-llamascope-slimpj-res-32k"
    cache_dir = f"caches/{model_name}/{sourceset}"
    
    refusal_cache = load_cache_file(f"{cache_dir}/refusal_cache.json")
    top_features_cache = load_cache_file(f"{cache_dir}/top_features_cache.json")
    
    if not refusal_cache or not top_features_cache:
        print("❌ Missing cache files")
        return
    
    feature_id = "18677"
    
    # Find all prompts where Feature 18677 appears
    feature_prompts = []
    
    for prompt, features in top_features_cache.items():
        for feature in features:
            if feature['featureIndex'] == feature_id:
                feature_prompts.append({
                    'prompt': prompt,
                    'activation': feature['activationValue'],
                    'refused': refusal_cache.get(prompt, False)
                })
                break
    
    if not feature_prompts:
        print(f"Feature {feature_id} not found in cache data.")
        return
    
    print(f"Feature {feature_id} appears in {len(feature_prompts)} prompts:")
    
    # Separate refused and complied prompts
    refused_prompts = [p for p in feature_prompts if p['refused']]
    complied_prompts = [p for p in feature_prompts if not p['refused']]
    
    print(f"  - Refused prompts: {len(refused_prompts)}")
    print(f"  - Complied prompts: {len(complied_prompts)}")
    
    # Calculate statistics
    if refused_prompts and complied_prompts:
        refused_activations = [p['activation'] for p in refused_prompts]
        complied_activations = [p['activation'] for p in complied_prompts]
        
        avg_refused = sum(refused_activations) / len(refused_activations)
        avg_complied = sum(complied_activations) / len(complied_activations)
        
        print(f"\n📈 ACTIVATION STATISTICS:")
        print(f"  - Average activation on refused prompts: {avg_refused:.4f}")
        print(f"  - Average activation on complied prompts: {avg_complied:.4f}")
        print(f"  - Difference: {avg_refused - avg_complied:.4f}")
        
        # Find highest activations
        max_refused = max(refused_activations)
        max_complied = max(complied_activations)
        
        print(f"  - Max activation on refused: {max_refused:.4f}")
        print(f"  - Max activation on complied: {max_complied:.4f}")
    
    # Show examples of high activations
    print(f"\n🔥 HIGHEST ACTIVATIONS:")
    print("-" * 30)
    
    # Sort by activation value
    feature_prompts.sort(key=lambda x: x['activation'], reverse=True)
    
    print("Top 10 highest activations:")
    for i, prompt_data in enumerate(feature_prompts[:10], 1):
        status = "🔴 REFUSED" if prompt_data['refused'] else "🟢 COMPLIED"
        print(f"{i:2d}. {status} - Activation: {prompt_data['activation']:.4f}")
        print(f"    Prompt: {prompt_data['prompt'][:80]}...")
        print()
    
    # Analyze prompt content patterns
    print(f"\n🔍 CONTENT ANALYSIS:")
    print("-" * 30)
    
    # Look for common words/phrases in high-activation prompts
    high_activation_prompts = [p for p in feature_prompts if p['activation'] > 3.0]
    
    if high_activation_prompts:
        print(f"Prompts with activation > 3.0: {len(high_activation_prompts)}")
        
        # Count common words
        word_counts = defaultdict(int)
        for prompt_data in high_activation_prompts:
            words = prompt_data['prompt'].lower().split()
            for word in words:
                if len(word) > 4:  # Only count longer words
                    word_counts[word] += 1
        
        # Show most common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nMost common words in high-activation prompts:")
        for word, count in common_words:
            print(f"  '{word}': {count} times")
    
    # Look for specific patterns
    print(f"\n🎯 PATTERN ANALYSIS:")
    print("-" * 30)
    
    # Check for specific keywords
    keywords = ['hack', 'bomb', 'illegal', 'fraud', 'steal', 'kill', 'drug', 'weapon', 'exploit']
    
    for keyword in keywords:
        matching_prompts = [p for p in feature_prompts if keyword.lower() in p['prompt'].lower()]
        if matching_prompts:
            avg_activation = sum(p['activation'] for p in matching_prompts) / len(matching_prompts)
            refused_count = sum(1 for p in matching_prompts if p['refused'])
            print(f"'{keyword}': {len(matching_prompts)} prompts, avg activation: {avg_activation:.4f}, {refused_count} refused")
    
    # Save detailed analysis
    analysis_results = {
        'feature_id': feature_id,
        'total_prompts': len(feature_prompts),
        'refused_prompts': len(refused_prompts),
        'complied_prompts': len(complied_prompts),
        'statistics': {
            'avg_refused_activation': avg_refused if refused_prompts else 0,
            'avg_complied_activation': avg_complied if complied_prompts else 0,
            'max_refused_activation': max_refused if refused_prompts else 0,
            'max_complied_activation': max_complied if complied_prompts else 0,
        },
        'top_activations': feature_prompts[:20],
        'high_activation_prompts': high_activation_prompts
    }
    
    output_file = f"feature_18677_detailed_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n📁 Detailed analysis saved to: {output_file}")

if __name__ == "__main__":
    investigate_feature_18677() 