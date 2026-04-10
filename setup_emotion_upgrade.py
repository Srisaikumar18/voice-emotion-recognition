"""
Setup script for emotion detection upgrade
Checks dependencies and guides through installation
"""
import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False
    except (ValueError, Exception) as e:
        # Package exists but has compatibility issues
        print(f"   ⚠️  Warning: {package_name} has compatibility issues: {str(e)[:50]}")
        return True  # Consider it installed but with issues

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("="*60)
    print("EMOTION DETECTION UPGRADE - SETUP")
    print("="*60)
    print()
    
    # Check Python version
    if not check_python_version():
        return
    print()
    
    # Check required packages
    packages = {
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib',
        'flask': 'Flask',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    print("Checking dependencies...")
    print("-" * 60)
    
    missing_packages = []
    for import_name, package_name in packages.items():
        if check_package(import_name):
            print(f"✅ {package_name}")
        else:
            print(f"❌ {package_name} - NOT INSTALLED")
            missing_packages.append(package_name)
    
    print()
    
    # Install missing packages
    if missing_packages:
        print(f"Found {len(missing_packages)} missing package(s)")
        print()
        
        response = input("Install missing packages? (y/n): ").lower()
        if response == 'y':
            print()
            print("Installing packages...")
            print("-" * 60)
            
            for package in missing_packages:
                print(f"Installing {package}...")
                if install_package(package):
                    print(f"✅ {package} installed successfully")
                else:
                    print(f"❌ Failed to install {package}")
            
            print()
            print("Installation complete!")
        else:
            print()
            print("Skipping installation. Install manually with:")
            print(f"  pip install {' '.join(missing_packages)}")
    else:
        print("✅ All dependencies are installed!")
    
    print()
    print("="*60)
    print("NEXT STEPS")
    print("="*60)
    print()
    
    # Check if data exists
    if os.path.exists("data/ravdess"):
        print("✅ RAVDESS dataset found")
    else:
        print("⚠️  RAVDESS dataset not found at data/ravdess/")
        print("   Make sure your dataset is in the correct location")
    
    print()
    
    # Check if models exist
    has_traditional = os.path.exists("model/emotion_model.pkl")
    has_deep = os.path.exists("model/emotion_cnn_model.h5")
    
    if has_traditional:
        print("✅ Traditional model found")
    else:
        print("⚠️  Traditional model not found")
    
    if has_deep:
        print("✅ Deep learning model found")
    else:
        print("⚠️  Deep learning model not found")
    
    print()
    
    if not has_deep:
        print("To train the deep learning model:")
        print("  python model/train_deep_model.py")
        print()
    
    print("To compare models:")
    print("  python model/compare_models.py")
    print()
    
    print("To start the application:")
    print("  python app.py")
    print()
    
    print("For detailed instructions, see:")
    print("  QUICK_START_EMOTION_UPGRADE.md")
    print()
    
    print("="*60)
    print("Setup check complete!")
    print("="*60)

if __name__ == "__main__":
    main()
