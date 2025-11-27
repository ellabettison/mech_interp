#!/usr/bin/env python3
"""
Setup script for cross-model transfer testing.
Checks API keys and provides guidance for configuration.
"""

import os
import sys
from typing import Dict, List, Optional

def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are available."""
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")
    }
    
    available = {}
    for key_name, key_value in api_keys.items():
        available[key_name] = key_value is not None and len(key_value.strip()) > 0
    
    return available

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    dependencies = {}
    
    try:
        import requests
        dependencies["requests"] = True
    except ImportError:
        dependencies["requests"] = False
    
    try:
        import dotenv
        dependencies["python-dotenv"] = True
    except ImportError:
        dependencies["python-dotenv"] = False
    
    return dependencies

def check_test_files() -> Dict[str, bool]:
    """Check if required test files exist."""
    files = {}
    
    files["api_rewrite_results.json"] = os.path.exists("api_rewrite_results.json")
    files["cross_model_transfer_tester.py"] = os.path.exists("cross_model_transfer_tester.py")
    files["test_cross_model_transfer.py"] = os.path.exists("test_cross_model_transfer.py")
    
    return files

def print_setup_status():
    """Print the current setup status."""
    print("🔧 CROSS-MODEL TRANSFER TESTING SETUP")
    print("=" * 50)
    
    # Check API keys
    print("\n📋 API KEY STATUS:")
    api_keys = check_api_keys()
    total_available = sum(api_keys.values())
    
    for key_name, is_available in api_keys.items():
        status = "✅ Available" if is_available else "❌ Missing"
        print(f"   {key_name}: {status}")
    
    print(f"\n   Total available: {total_available}/{len(api_keys)}")
    
    # Check dependencies
    print("\n📦 DEPENDENCY STATUS:")
    dependencies = check_dependencies()
    total_installed = sum(dependencies.values())
    
    for dep_name, is_installed in dependencies.items():
        status = "✅ Installed" if is_installed else "❌ Missing"
        print(f"   {dep_name}: {status}")
    
    print(f"\n   Total installed: {total_installed}/{len(dependencies)}")
    
    # Check test files
    print("\n📁 FILE STATUS:")
    files = check_test_files()
    total_files = sum(files.values())
    
    for file_name, exists in files.items():
        status = "✅ Found" if exists else "❌ Missing"
        print(f"   {file_name}: {status}")
    
    print(f"\n   Total files: {total_files}/{len(files)}")
    
    return api_keys, dependencies, files

def provide_guidance(api_keys: Dict[str, bool], dependencies: Dict[str, bool], files: Dict[str, bool]):
    """Provide guidance based on current setup status."""
    
    print("\n💡 SETUP GUIDANCE:")
    
    # API Key guidance
    if not any(api_keys.values()):
        print("\n🔑 API KEYS NEEDED:")
        print("   To test real models, you need at least one API key:")
        print("   - OPENAI_API_KEY: For GPT-4, GPT-3.5")
        print("   - ANTHROPIC_API_KEY: For Claude models")
        print("   - TOGETHER_API_KEY: For Llama, Mistral models (recommended)")
        print("   - HF_TOKEN or HUGGINGFACE_TOKEN: For GPT-2 models (fallback)")
        print("   - GEMINI_API_KEY: For Gemini models")
        print("\n   Set them as environment variables:")
        print("   export OPENAI_API_KEY='your-key-here'")
    else:
        print("\n✅ API KEYS READY:")
        available_models = []
        if api_keys["OPENAI_API_KEY"]:
            available_models.extend(["GPT-4o-mini", "GPT-3.5-turbo"])
        if api_keys["ANTHROPIC_API_KEY"]:
            available_models.extend(["Claude-3-5-Sonnet", "Claude-3-Haiku"])
        if api_keys["TOGETHER_API_KEY"]:
            available_models.extend(["Llama-2-7b", "Mistral-7b", "Mixtral-8x7b"])
        if api_keys["HF_TOKEN"]:
            available_models.extend(["GPT-2", "DistilGPT-2"])
        if api_keys["GEMINI_API_KEY"]:
            available_models.extend(["Gemini-2.0-Flash"])
        
        print(f"   Available models: {', '.join(available_models)}")
    
    # Dependency guidance
    missing_deps = [name for name, installed in dependencies.items() if not installed]
    if missing_deps:
        print(f"\n📦 INSTALL MISSING DEPENDENCIES:")
        print("   pip install " + " ".join(missing_deps))
    else:
        print("\n✅ DEPENDENCIES READY")
    
    # File guidance
    missing_files = [name for name, exists in files.items() if not exists]
    if missing_files:
        print(f"\n📁 MISSING FILES:")
        for file_name in missing_files:
            if file_name == "api_rewrite_results.json":
                print("   - Run api_based_rewriter.py first to generate test prompts")
            else:
                print(f"   - {file_name} is missing")
    else:
        print("\n✅ TEST FILES READY")
    
    # Next steps
    print("\n🚀 NEXT STEPS:")
    if all(dependencies.values()) and all(files.values()):
        if any(api_keys.values()):
            print("   1. ✅ Ready for real testing!")
            print("   2. Run: python3 cross_model_transfer_tester.py")
        else:
            print("   1. ✅ Ready for mock testing!")
            print("   2. Run: python3 test_cross_model_transfer.py")
            print("   3. Add API keys for real model testing")
    else:
        print("   1. Install missing dependencies")
        print("   2. Generate test prompts (run api_based_rewriter.py)")
        print("   3. Add API keys for model testing")

def test_mock_run():
    """Test if the mock system can run."""
    print("\n🧪 TESTING MOCK SYSTEM:")
    try:
        import test_cross_model_transfer
        print("   ✅ Mock test script is ready")
        return True
    except ImportError as e:
        print(f"   ❌ Mock test script error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Cross-Model Transfer Testing Setup")
    print("=" * 50)
    
    # Check current status
    api_keys, dependencies, files = print_setup_status()
    
    # Provide guidance
    provide_guidance(api_keys, dependencies, files)
    
    # Test mock system
    mock_ready = test_mock_run()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SETUP SUMMARY:")
    
    total_ready = sum([
        any(api_keys.values()),  # At least one API key
        all(dependencies.values()),  # All dependencies
        all(files.values()),  # All files
        mock_ready  # Mock system works
    ])
    
    if total_ready >= 3:
        print("   ✅ Ready for testing!")
        if any(api_keys.values()):
            print("   🎯 You can run real cross-model transfer tests")
        else:
            print("   🎯 You can run mock tests (add API keys for real testing)")
    elif total_ready >= 2:
        print("   ⚠️  Almost ready - check missing items above")
    else:
        print("   ❌ Setup incomplete - follow guidance above")
    
    print(f"   📈 Setup completeness: {total_ready}/4 components ready")
    print("=" * 50)

if __name__ == "__main__":
    main() 