# Installation Checklist - Emotion Detection Upgrade

## Pre-Installation Check

- [ ] Python 3.8+ installed
- [ ] RAVDESS dataset in `data/ravdess/` folder
- [ ] At least 4GB free RAM
- [ ] Internet connection (for package installation)

## Step 1: Check Current Setup

Run the setup checker:
```bash
python setup_emotion_upgrade.py
```

This will tell you:
- ✅ What's already installed
- ❌ What's missing
- 📋 Next steps

## Step 2: Install Dependencies

### Option A: Automatic (Recommended)
```bash
python setup_emotion_upgrade.py
# Follow the prompts to install missing packages
```

### Option B: Manual
```bash
pip install tensorflow>=2.10.0
pip install -r requirements_deep.txt
```

### Option C: CPU-only TensorFlow (Faster installation)
```bash
pip install tensorflow-cpu>=2.10.0
pip install -r requirements_deep.txt
```

**Verify installation:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
```

Expected output: `TensorFlow 2.x.x installed`

## Step 3: Train the Deep Learning Model

```bash
python model/train_deep_model.py
```

**What to expect:**
- Duration: 10-15 minutes (CPU) or 3-5 minutes (GPU)
- Output: Progress bars showing training epochs
- Final accuracy: Should be 80%+

**Success indicators:**
```
✅ "Extracted features from 1440 files"
✅ "Test Accuracy: 0.85+" (85% or higher)
✅ "Model saved to model/emotion_cnn_model.h5"
```

**If training fails:**
- Check that `data/ravdess/` exists and contains audio files
- Reduce batch size if out of memory
- Check console for specific error messages

## Step 4: Compare Models (Optional but Recommended)

```bash
python model/compare_models.py
```

**What to expect:**
- Comparison of traditional vs deep learning model
- Accuracy metrics for both
- Improvement percentage

**Success indicators:**
```
✅ Traditional Model Accuracy: 60-70%
✅ Deep Learning Model Accuracy: 80-90%
✅ Improvement: +15-25 percentage points
```

## Step 5: Test in Application

### Start the app:
```bash
python app.py
```

### Test with uploaded audio:
1. Open browser to `http://localhost:5000`
2. Upload an audio file with clear emotion
3. Check the prediction

**Success indicators:**
- ✅ Emotion is predicted correctly
- ✅ Confidence score is shown (e.g., "angry (87.3%)")
- ✅ High confidence (80%+) for clear emotions

## Verification Checklist

After installation, verify everything works:

### Dependencies
- [ ] TensorFlow installed and importable
- [ ] All packages from `requirements_deep.txt` installed
- [ ] No import errors when running `python setup_emotion_upgrade.py`

### Models
- [ ] `model/emotion_cnn_model.h5` exists (deep learning model)
- [ ] `model/label_encoder.pkl` exists (label encoder)
- [ ] `model/feature_params.pkl` exists (feature parameters)
- [ ] `model/emotion_model.pkl` exists (traditional model - fallback)

### Application
- [ ] App starts without errors (`python app.py`)
- [ ] Can upload audio files
- [ ] Emotion predictions include confidence scores
- [ ] Predictions are accurate for clear emotions

### Performance
- [ ] Training completed in reasonable time (< 20 minutes)
- [ ] Test accuracy is 80%+ 
- [ ] Inference time is < 1 second per audio file
- [ ] No memory errors during prediction

## Troubleshooting

### Issue: "TensorFlow not found"
**Solution:**
```bash
pip install tensorflow-cpu
```

### Issue: "No module named 'keras'"
**Solution:**
```bash
pip install tensorflow>=2.10.0
# Keras is included in TensorFlow 2.x
```

### Issue: "Out of memory during training"
**Solution:** Edit `model/train_deep_model.py`:
```python
# Line ~280, reduce batch_size
history = model.fit(
    X_train, y_train_cat,
    batch_size=16,  # Change from 32 to 16
    ...
)
```

### Issue: "Model accuracy is low (< 70%)"
**Possible causes:**
1. Dataset issues - check RAVDESS files
2. Insufficient training - increase epochs
3. Imbalanced data - check class distribution

**Solution:**
```bash
# Check data distribution
python -c "from model.train_deep_model import *; import os; files = [f for root, _, files in os.walk(DATA_PATH) for f in files if f.endswith('.wav')]; print(f'Total files: {len(files)}')"
```

### Issue: "App uses old model"
**Check:**
1. Is `model/emotion_cnn_model.h5` present?
2. Is TensorFlow installed?
3. Check console output when starting app

**Solution:**
```bash
# Verify model exists
ls model/emotion_cnn_model.h5

# Verify TensorFlow
python -c "import tensorflow; print('TensorFlow OK')"
```

## Quick Reference

### File Locations
```
model/
├── emotion_cnn_model.h5      # Deep learning model (NEW)
├── label_encoder.pkl          # Label encoder (NEW)
├── feature_params.pkl         # Feature parameters (NEW)
├── emotion_model.pkl          # Traditional model (FALLBACK)
├── train_deep_model.py        # Training script (NEW)
└── compare_models.py          # Comparison tool (NEW)

audio_utils/
└── preprocess_deep.py         # Feature extraction (NEW)

app.py                         # Updated with DL support
```

### Commands
```bash
# Check setup
python setup_emotion_upgrade.py

# Train model
python model/train_deep_model.py

# Compare models
python model/compare_models.py

# Start app
python app.py
```

### Documentation
- **Quick Start**: `QUICK_START_EMOTION_UPGRADE.md`
- **Detailed Guide**: `EMOTION_DETECTION_UPGRADE.md`
- **Summary**: `EMOTION_UPGRADE_SUMMARY.md`
- **This Checklist**: `INSTALLATION_CHECKLIST.md`

## Success Criteria

Your installation is successful when:

1. ✅ All dependencies installed without errors
2. ✅ Deep learning model trained with 80%+ accuracy
3. ✅ Application starts and loads deep learning model
4. ✅ Predictions include confidence scores
5. ✅ Accuracy is noticeably better than before

## Next Steps After Installation

1. **Test with various audio files**
   - Try different emotions
   - Try different speakers
   - Try different audio quality

2. **Monitor performance**
   - Check confidence scores
   - Note which emotions are most accurate
   - Identify areas for improvement

3. **Fine-tune if needed**
   - Adjust training parameters
   - Add more training data
   - Experiment with model architecture

4. **Deploy to production**
   - Test thoroughly
   - Monitor in real-world use
   - Collect feedback

## Support

If you encounter issues:

1. Check this checklist
2. Review `EMOTION_DETECTION_UPGRADE.md`
3. Run `python setup_emotion_upgrade.py` for diagnostics
4. Check console output for error messages

## Completion

When all checkboxes are marked:
- ✅ Installation complete!
- ✅ Ready for production use!
- ✅ Enjoy your upgraded emotion detection system! 🎉

---

**Current Status**: Ready to install
**Estimated Time**: 20-30 minutes total
**Difficulty**: Easy (automated scripts provided)
