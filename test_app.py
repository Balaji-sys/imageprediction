"""
Test script to verify the Flask app loads correctly
"""
import sys
import os

# Add the project directory to path
sys.path.insert(0, os.getcwd())

try:
    from app import app, model, class_names
    print("✓ App imported successfully")
    print(f"✓ Model loaded: {model.input_shape} -> {model.output_shape}")
    print(f"✓ Number of classes: {len(class_names)}")
    print(f"✓ Classes: {list(class_names.values())[:3]}... (showing first 3)")
    print("\n✓✓✓ APP IS READY TO RUN ✓✓✓")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
