# Manual Installation Guide

## Problem

You're seeing this error:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

This is a numpy/pandas version mismatch. Here's how to fix it.

## Quick Fix (Recommended)

Run the automated fix script:

```bash
python fix_dependencies.py
```

This will:
1. Fix numpy/pandas compatibility
2. Install TensorFlow
3. Verify everything works

## Manual Fix (If automated script fails)

### Step 1: Fix numpy/pandas compatibility

```bash
pip uninstall -y numpy pandas scikit-learn
pip install numpy==1.23.5
pip install pandas scikit-learn
```

### Step 2: Install TensorFlow

**Option A: CPU version (Recommended - faster install, smaller size)**
```bash
pip install tensorflow-cpu==2.13.0
```

**Option B: Full version (if you have GPU)**
```bash
pip install tensorflow==2.13.0
```

### Step 3: Verify installation

```bash
python -c "import numpy; print(f'numpy {numpy.__version__}')"
python -c "import pandas; print(f'pandas OK')"
python -c "import sklearn; print(f'sklearn OK')"
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

Expected output:
```
numpy 1.23.5
pandas OK
sklearn OK
TensorFlow 2.13.0
```

## Alternative: Use Virtual Environment (Cleanest approach)

If you want a clean installation without affecting other projects:

### Step 1: Create virtual environment

```bash
python -m venv emotion_env
```

### Step 2: Activate it

**Windows:**
```bash
emotion_env\Scripts\activate
```

**Linux/Mac:**
```bash
source emotion_env/bin/activate
```

### Step 3: Install everything fresh

```bash
pip install numpy==1.23.5
pip install pandas scikit-learn
pip install librosa soundfile
pip install Flask scipy matplotlib
pip install SpeechRecognition deep-translator noisereduce
pip install tensorflow-cpu==2.13.0
pip install joblib
```

### Step 4: Verify

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} ready!')"
```

## After Installation

Once dependencies are fixed, proceed with:

### 1. Train the model

```bash
python model/train_deep_model.py
```

Expected output:
```
Starting feature extraction with deep learning approach...
Extracted features from 1440 files.
...
Test Accuracy: 0.85+
Model saved to model/emotion_cnn_model.h5
```

### 2. Test the app

```bash
python app.py
```

Then open `http://localhost:5000` and upload audio files.

## Troubleshooting

### Issue: "pip is not recognized"

**Solution:**
```bash
python -m pip install <package>
```

### Issue: "Permission denied"

**Solution (Windows):**
```bash
pip install --user <package>
```

### Issue: TensorFlow installation is very slow

**Solution:** Use CPU version
```bash
pip install tensorflow-cpu
```

### Issue: Still getting numpy errors

**Solution:** Force reinstall
```bash
pip uninstall -y numpy
pip cache purge
pip install numpy==1.23.5 --no-cache-dir
```

### Issue: "Could not find a version that satisfies the requirement"

**Solution:** Update pip first
```bash
python -m pip install --upgrade pip
```

Then retry the installation.

## Verification Checklist

After installation, verify:

- [ ] `python -c "import numpy"` works
- [ ] `python -c "import pandas"` works
- [ ] `python -c "import sklearn"` works
- [ ] `python -c "import tensorflow"` works
- [ ] `python -c "import librosa"` works
- [ ] No error messages when importing

## Next Steps

Once all dependencies are installed:

1. ✅ Train the model: `python model/train_deep_model.py`
2. ✅ Start the app: `python app.py`
3. ✅ Upload audio files and test predictions

## Need More Help?

If you're still having issues:

1. Check your Python version: `python --version` (should be 3.8+)
2. Check pip version: `pip --version` (should be 20.0+)
3. Try the virtual environment approach (cleanest solution)
4. Share the specific error message for more targeted help

## Summary

The issue is a numpy/pandas version mismatch. The fix is:

```bash
# Quick fix
python fix_dependencies.py

# Or manual fix
pip uninstall -y numpy pandas scikit-learn
pip install numpy==1.23.5 pandas scikit-learn tensorflow-cpu
```

Then train and test:
```bash
python model/train_deep_model.py
python app.py
```

That's it! 🎉
