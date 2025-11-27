#!/usr/bin/env python3
"""
Script to analyze whether rhetorical questions (Feature 9558) are refused more often.
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

def analyze_rhetorical_questions():
    """Analyze whether rhetorical questions are refused more often."""
    
    print("🔍 ANALYZING RHETORICAL QUESTIONS (FEATURE 9558)")
    print("=" * 60)
    
    # Load cache data
    model_name = "deepseek-r1-distill-llama-8b"
    sourceset = "15-llamascope-slimpj-res-32k"
    cache_dir = f"caches/{model_name}/{sourceset}"
    
    refusal_cache = load_cache_file(f"{cache_dir}/refusal_cache.json")
    top_features_cache = load_cache_file(f"{cache_dir}/top_features_cache.json")
    
    if not refusal_cache or not top_features_cache:
        print("❌ Missing cache files")
        return
    
    feature_id = "9558"  # Rhetorical questions feature
    
    # Find all prompts where Feature 9558 appears
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
    
    print(f"Feature 9558 appears in {len(feature_prompts)} prompts")
    
    # Separate refused and complied prompts
    refused_prompts = [p for p in feature_prompts if p['refused']]
    complied_prompts = [p for p in feature_prompts if not p['refused']]
    
    print(f"\n📊 RHETORICAL QUESTIONS ANALYSIS:")
    print(f"  - Total prompts with rhetorical questions: {len(feature_prompts)}")
    print(f"  - Refused prompts: {len(refused_prompts)} ({len(refused_prompts)/len(feature_prompts)*100:.1f}%)")
    print(f"  - Complied prompts: {len(complied_prompts)} ({len(complied_prompts)/len(feature_prompts)*100:.1f}%)")
    
    # Compare with overall refusal rate
    total_prompts = len(refusal_cache)
    total_refused = sum(1 for refused in refusal_cache.values() if refused)
    overall_refusal_rate = total_refused / total_prompts
    
    print(f"\n📈 COMPARISON WITH OVERALL RATE:")
    print(f"  - Overall refusal rate: {overall_refusal_rate*100:.1f}%")
    print(f"  - Rhetorical questions refusal rate: {len(refused_prompts)/len(feature_prompts)*100:.1f}%")
    
    if len(refused_prompts)/len(feature_prompts) > overall_refusal_rate:
        print(f"  ✅ Rhetorical questions are refused MORE often than average")
    else:
        print(f"  ❌ Rhetorical questions are refused LESS often than average")
    
    # Analyze activation levels
    refused_activations = [p['activation'] for p in refused_prompts]
    complied_activations = [p['activation'] for p in complied_prompts]
    
    avg_refused = sum(refused_activations) / len(refused_activations) if refused_activations else 0
    avg_complied = sum(complied_activations) / len(complied_activations) if complied_activations else 0
    
    print(f"\n⚡ ACTIVATION ANALYSIS:")
    print(f"  - Average activation on refused prompts: {avg_refused:.4f}")
    print(f"  - Average activation on complied prompts: {avg_complied:.4f}")
    print(f"  - Difference: {avg_refused - avg_complied:.4f}")
    
    # Look at high activation prompts specifically
    high_activation_threshold = 3.0
    high_activation_prompts = [p for p in feature_prompts if p['activation'] > high_activation_threshold]
    high_refused = [p for p in high_activation_prompts if p['refused']]
    high_complied = [p for p in high_activation_prompts if not p['refused']]
    
    print(f"\n🔥 HIGH ACTIVATION ANALYSIS (>3.0):")
    print(f"  - High activation prompts: {len(high_activation_prompts)}")
    print(f"  - High activation refused: {len(high_refused)} ({len(high_refused)/len(high_activation_prompts)*100:.1f}%)")
    print(f"  - High activation complied: {len(high_complied)} ({len(high_complied)/len(high_activation_prompts)*100:.1f}%)")
    
    # Show examples of high activation prompts
    print(f"\n📝 EXAMPLES OF HIGH ACTIVATION PROMPTS:")
    print("-" * 40)
    
    high_activation_prompts.sort(key=lambda x: x['activation'], reverse=True)
    
    for i, prompt_data in enumerate(high_activation_prompts[:10], 1):
        status = "🔴 REFUSED" if prompt_data['refused'] else "🟢 COMPLIED"
        print(f"{i:2d}. {status} - Activation: {prompt_data['activation']:.4f}")
        print(f"    {prompt_data['prompt'][:100]}...")
        print()
    
    # Look for question patterns
    print(f"\n❓ QUESTION PATTERN ANALYSIS:")
    print("-" * 40)
    
    question_keywords = ['how', 'what', 'why', 'when', 'where', 'who', 'which', '?']
    question_prompts = []
    
    for prompt_data in feature_prompts:
        prompt_lower = prompt_data['prompt'].lower()
        if any(keyword in prompt_lower for keyword in question_keywords):
            question_prompts.append(prompt_data)
    
    if question_prompts:
        question_refused = [p for p in question_prompts if p['refused']]
        print(f"  - Prompts with question keywords: {len(question_prompts)}")
        print(f"  - Question prompts refused: {len(question_refused)} ({len(question_refused)/len(question_prompts)*100:.1f}%)")
        print(f"  - Question prompts complied: {len(question_prompts) - len(question_refused)} ({(len(question_prompts) - len(question_refused))/len(question_prompts)*100:.1f}%)")
    
    # Statistical significance
    print(f"\n📊 STATISTICAL SIGNIFICANCE:")
    print("-" * 40)
    
    # Chi-square test for independence
    from scipy.stats import chi2_contingency
    import numpy as np
    
    # Create contingency table
    # Rows: Feature 9558 present/absent, Columns: Refused/Complied
    feature_present_refused = len(refused_prompts)
    feature_present_complied = len(complied_prompts)
    feature_absent_refused = total_refused - feature_present_refused
    feature_absent_complied = (total_prompts - total_refused) - feature_present_complied
    
    contingency_table = np.array([
        [feature_present_refused, feature_present_complied],
        [feature_absent_refused, feature_absent_complied]
    ])
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"  - Chi-square statistic: {chi2:.4f}")
        print(f"  - P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✅ Statistically significant relationship (p < 0.05)")
        else:
            print(f"  ❌ No statistically significant relationship (p >= 0.05)")
            
    except Exception as e:
        print(f"  ❌ Could not calculate statistical significance: {e}")

if __name__ == "__main__":
    analyze_rhetorical_questions() 