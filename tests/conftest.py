"""
Root conftest.py to configure pytest for all tests.
"""
import sys
import os
from pathlib import Path

# Add the src directory to Python's module search path
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
    print(f"Added {src_dir} to Python path")

# Print debugging info
print(f"Project root: {project_root}")
print(f"Source directory: {src_dir}")
print(f"Python path: {sys.path}")

# Check for src/chuk_ai_session_manager directory
package_dir = src_dir / "chuk_ai_session_manager"
if not package_dir.exists():
    print(f"WARNING: Package directory not found: {package_dir}")
else:
    print(f"Package directory found: {package_dir}")
    
    # Check for key modules
    models_dir = package_dir / "models"
    storage_dir = package_dir / "storage"
    
    if models_dir.exists():
        print(f"Models directory found: {models_dir}")
        model_files = list(models_dir.glob("*.py"))
        print(f"Model files: {[f.name for f in model_files]}")
    
    if storage_dir.exists():
        print(f"Storage directory found: {storage_dir}")
        storage_files = list(storage_dir.glob("*.py"))
        print(f"Storage files: {[f.name for f in storage_files]}")
