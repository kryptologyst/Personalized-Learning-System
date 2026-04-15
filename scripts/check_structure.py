#!/usr/bin/env python3
"""Simple structure verification script."""

import os
import sys
from pathlib import Path

def check_structure():
    """Check if the project structure is correct."""
    
    print("🎓 Personalized Learning System - Structure Verification")
    print("=" * 60)
    
    # Check main directories
    directories = [
        "src", "src/data", "src/models", "src/eval", "src/viz",
        "configs", "data", "data/raw", "data/processed", "data/external",
        "scripts", "tests", "demo", "assets", "notebooks"
    ]
    
    print("\n📁 Directory Structure:")
    all_good = True
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ - MISSING")
            all_good = False
    
    # Check main files
    files = [
        "README.md", "requirements.txt", "pyproject.toml", ".gitignore",
        "DISCLAIMER.md", "src/__init__.py", "src/data/__init__.py",
        "src/models/__init__.py", "src/eval/__init__.py", "src/viz/__init__.py",
        "src/data/data_generator.py", "src/models/models.py", "src/eval/evaluator.py",
        "src/viz/visualizer.py", "scripts/train.py", "scripts/demo.py",
        "demo/app.py", "tests/test_system.py", "configs/config.yaml",
        "configs/data.yaml", ".github/workflows/ci.yml"
    ]
    
    print("\n📄 Key Files:")
    for file in files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_good = False
    
    # Check file sizes
    print("\n📊 File Sizes:")
    key_files = [
        "README.md", "src/data/data_generator.py", "src/models/models.py",
        "src/eval/evaluator.py", "demo/app.py"
    ]
    
    for file in key_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  {file}: {size:,} bytes")
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 All structure checks passed! Project is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run demo: python3 scripts/demo.py")
        print("3. Launch Streamlit: streamlit run demo/app.py")
        print("4. Run tests: pytest tests/")
    else:
        print("⚠️  Some structure issues found. Please check missing files/directories.")
    
    return all_good


if __name__ == "__main__":
    check_structure()
