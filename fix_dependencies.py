"""
Fix numpy/pandas compatibility issues and install TensorFlow
"""
import subprocess
import sys

def run_command(command, description):
    """Run a pip command and report results"""
    print(f"\n{description}...")
    print("-" * 60)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Success")
            return True
        else:
            print(f"❌ Failed: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*60)
    print("DEPENDENCY FIX SCRIPT")
    print("="*60)
    print("\nThis will fix numpy/pandas compatibility and install TensorFlow")
    print()
    
    response = input("Continue? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print("\n" + "="*60)
    print("STEP 1: Fix numpy/pandas compatibility")
    print("="*60)
    
    # Uninstall and reinstall numpy and pandas to fix compatibility
    commands = [
        (f'"{sys.executable}" -m pip uninstall -y numpy pandas', 
         "Uninstalling numpy and pandas"),
        (f'"{sys.executable}" -m pip install numpy==1.23.5', 
         "Installing numpy 1.23.5"),
        (f'"{sys.executable}" -m pip install pandas', 
         "Reinstalling pandas"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            print("\n⚠️  Warning: Some commands failed, but continuing...")
    
    print("\n" + "="*60)
    print("STEP 2: Install TensorFlow")
    print("="*60)
    
    # Try TensorFlow CPU first (faster, more compatible)
    if not run_command(
        f'"{sys.executable}" -m pip install tensorflow-cpu>=2.10.0',
        "Installing TensorFlow (CPU version)"
    ):
        print("\nTrying full TensorFlow package...")
        run_command(
            f'"{sys.executable}" -m pip install tensorflow>=2.10.0',
            "Installing TensorFlow (full version)"
        )
    
    print("\n" + "="*60)
    print("STEP 3: Verify installation")
    print("="*60)
    
    # Test imports
    print("\nTesting imports...")
    
    packages = ['numpy', 'pandas', 'sklearn', 'tensorflow']
    all_ok = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except Exception as e:
            print(f"❌ {package}: {str(e)[:50]}")
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Train the model: python model/train_deep_model.py")
        print("2. Start the app: python app.py")
    else:
        print("⚠️  SOME ISSUES REMAIN")
        print("="*60)
        print("\nTry manual installation:")
        print("  pip uninstall -y numpy pandas scikit-learn")
        print("  pip install numpy==1.23.5")
        print("  pip install pandas scikit-learn")
        print("  pip install tensorflow-cpu")

if __name__ == "__main__":
    main()
