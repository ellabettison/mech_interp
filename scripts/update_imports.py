#!/usr/bin/env python3
"""
Script to update import statements after reorganization.
This script helps fix import paths that may have been broken during the reorganization.
"""

import os
import re
import glob
from pathlib import Path

def update_imports_in_file(file_path):
    """Update import statements in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Common import patterns that might need updating
    import_patterns = [
        # Update relative imports for moved modules
        (r'from \.\.auto_feature_finder', 'from src.utils.auto_feature_finder'),
        (r'from \.\.model_calling', 'from src.utils.model_calling'),
        (r'from \.\.api', 'from src.api'),
        (r'from \.\.analysis', 'from src.analysis'),
        (r'from \.\.rewriters', 'from src.rewriters'),
        (r'from \.\.core', 'from src.core'),
        
        # Update absolute imports
        (r'import auto_feature_finder', 'import src.utils.auto_feature_finder'),
        (r'import model_calling', 'import src.utils.model_calling'),
        (r'import api', 'import src.api'),
        (r'import analysis', 'import src.analysis'),
        (r'import rewriters', 'import src.rewriters'),
        (r'import core', 'import src.core'),
    ]
    
    for pattern, replacement in import_patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Updated imports in {file_path}")
        return True
    
    return False

def main():
    """Update imports in all Python files."""
    # Find all Python files in the src directory
    src_files = glob.glob('src/**/*.py', recursive=True)
    script_files = glob.glob('scripts/*.py', recursive=True)
    
    all_files = src_files + script_files
    
    updated_count = 0
    for file_path in all_files:
        if update_imports_in_file(file_path):
            updated_count += 1
    
    print(f"\nUpdated imports in {updated_count} files.")
    print("\nNote: You may need to manually review and update some import statements.")
    print("Common issues to check:")
    print("- Relative imports that reference moved modules")
    print("- Import statements that reference specific file paths")
    print("- Module-level imports that need to be updated")

if __name__ == "__main__":
    main() 