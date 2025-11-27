#!/usr/bin/env python3
"""
Script to check which HuggingFace models are available via the inference API.
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_model_availability(model_name, api_key):
    """Test if a model is available via the inference API."""
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Simple test prompt
    data = {
        "inputs": "Hello, how are you?",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.0,
            "do_sample": False
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return True, "Available"
        elif response.status_code == 404:
            return False, "Model not found"
        elif response.status_code == 403:
            return False, "Access denied (requires special access)"
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def main():
    """Test various model availability."""
    api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    if not api_key:
        print("❌ No HuggingFace token found. Set HF_TOKEN environment variable.")
        return
    
    print("🔍 Testing HuggingFace Model Availability")
    print("=" * 50)
    
    # Models to test
    models_to_test = [
        # Llama models
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "NousResearch/Llama-2-7b-chat-hf",
        "NousResearch/Llama-2-13b-chat-hf",
        
        # Mistral models
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        
        # Other popular models
        "microsoft/DialoGPT-medium",
        "gpt2",
        "distilgpt2",
        "EleutherAI/gpt-neo-125M",
        "EleutherAI/gpt-neo-1.3B",
        
        # Smaller, accessible models
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-large",
        "distilbert-base-uncased",
        "bert-base-uncased"
    ]
    
    available_models = []
    unavailable_models = []
    
    for model_name in models_to_test:
        print(f"\nTesting: {model_name}")
        is_available, message = test_model_availability(model_name, api_key)
        
        if is_available:
            print(f"   ✅ {message}")
            available_models.append(model_name)
        else:
            print(f"   ❌ {message}")
            unavailable_models.append(model_name)
    
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"\n✅ AVAILABLE MODELS ({len(available_models)}):")
    for model in available_models:
        print(f"   - {model}")
    
    print(f"\n❌ UNAVAILABLE MODELS ({len(unavailable_models)}):")
    for model in unavailable_models:
        print(f"   - {model}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if available_models:
        print("   Use these models for cross-model transfer testing:")
        for model in available_models[:5]:  # Show top 5
            print(f"     - {model}")
    else:
        print("   No models available. Check your API key and permissions.")

if __name__ == "__main__":
    main() 