#!/usr/bin/env python3
"""
Test script to verify R57 standalone package functionality
"""

import sys
import importlib
import os

def test_imports():
    """Test that all modules can be imported successfully"""
    print("🔍 Testing module imports...")
    
    modules_to_test = [
        'utils',
        'aligner', 
        'warp_utils',
        'warper',
        'foreign_object_utils',
        'deform_losses',
        'networks',
        'deform_image_by_electrode',
        'main_make_cylendrical_deformation_field'
    ]
    
    # Ensure current directory is in sys.path for local imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            return False
    
    return True

def test_main_function():
    """Test that the main deformation function can be imported"""
    print("\n🔍 Testing main function import...")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        module = importlib.import_module('deform_image_by_electrode')
        func = getattr(module, 'deform_image_by_electrode', None)
        if func is not None:
            print("  ✅ deform_image_by_electrode function imported successfully")
            return True
        else:
            print("  ❌ deform_image_by_electrode function not found")
            return False
    except ImportError as e:
        print(f"  ❌ Failed to import deform_image_by_electrode: {e}")
        return False

def test_external_dependencies():
    """Test that external dependencies are available"""
    print("\n🔍 Testing external dependencies...")
    
    external_deps = [
        'numpy',
        'nibabel', 
        'SimpleITK',
        'nilearn',
        'monai',
        'torch'
    ]
    
    for dep in external_deps:
        try:
            importlib.import_module(dep)
            print(f"  ✅ {dep}")
        except ImportError as e:
            print(f"  ❌ {dep}: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🚀 R57 Standalone Package Test Suite")
    print("=" * 50)
    
    tests = [
        test_external_dependencies,
        test_imports,
        test_main_function
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! R57 standalone package is ready to use.")
        print("\nNext steps:")
        print("  • Run notebook: jupyter lab main_reg_for_Seymour_Ryan_R57.ipynb")
        print("  • Run script: python main_deform_image_by_electrode_r57.py")
    else:
        print("❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
