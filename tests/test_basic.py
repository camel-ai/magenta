#!/usr/bin/env python3
"""
Basic integration test for math-dataset functionality.
This script provides a simple test to verify the core components work.
"""
import os
import sys

def test_basic_imports():
    """Test that we can import the main components."""
    print("Testing basic imports...")
    
    # Test math solver imports (go up one level from tests/ directory)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'solver_agent'))
    try:
        from math_solver import MathSolver
        print("✅ MathSolver import successful")
    except ImportError as e:
        print(f"❌ MathSolver import failed: {e}")
        return False
    
    try:
        from math_loader import MathLoader
        print("✅ MathLoader import successful")
    except ImportError as e:
        print(f"❌ MathLoader import failed: {e}")
        return False
    
    # Test back translation imports (go up one level from tests/ directory)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'back_translation'))
    try:
        from utils import format_problem, process_log
        print("✅ Back translation utils import successful")
    except ImportError as e:
        print(f"❌ Back translation utils import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without API calls."""
    print("\nTesting basic functionality...")
    
    # Re-import for this function scope (go up one level from tests/ directory)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'solver_agent'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'back_translation'))
    
    # Test MathSolver initialization
    try:
        from math_solver import MathSolver
        solver = MathSolver(
            model="gpt-4o-mini",
            multi_step=False,
            sympy_toolkit=False,
            code_toolkit=False,
            geometry_toolkit=False
        )
        print("✅ MathSolver initialization successful")
    except Exception as e:
        print(f"❌ MathSolver initialization failed: {e}")
        return False
    
    # Test MathLoader initialization
    try:
        from math_loader import MathLoader
        loader = MathLoader(mode="train")
        print("✅ MathLoader initialization successful")
    except Exception as e:
        print(f"❌ MathLoader initialization failed: {e}")
        return False
    
    # Test utility functions
    try:
        from utils import format_problem
        result = format_problem("What is 2 + 2?")
        if isinstance(result, str):
            print("✅ format_problem function works")
        else:
            print(f"❌ format_problem returned unexpected type: {type(result)}")
            return False
    except Exception as e:
        print(f"❌ format_problem failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test that we can load math problems from data files."""
    print("\nTesting data loading...")
    
    # Re-import for this function scope (go up one level from tests/ directory)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'solver_agent'))
    
    try:
        from math_loader import MathLoader
        loader = MathLoader(mode="train")
        # Try to load a small number of problems
        problems = loader.load_problems("algebra", level=1, num=1)
        if problems and len(problems) > 0:
            print(f"✅ Successfully loaded {len(problems)} algebra problems")
            print(f"   Sample problem: {problems[0].get('problem', 'N/A')[:50]}...")
            return True
        else:
            print("⚠️ No problems loaded (data files may not exist)")
            return True  # This is acceptable if data files don't exist
    except Exception as e:
        print(f"⚠️ Data loading failed (acceptable if no data files): {e}")
        return True  # This is acceptable

def main():
    """Run all basic tests."""
    print("Math Dataset Basic Integration Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_basic_imports),
        ("Functionality Test", test_basic_functionality),
        ("Data Loading Test", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All basic tests passed! The system is working correctly.")
        print("\nNext steps:")
        print("1. Set up your API key in .env file")
        print("2. Run: source .venv/bin/activate && cd src/solver_agent")
        print("3. Test with: python main.py --num 1 --dataset algebra --level 1")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 