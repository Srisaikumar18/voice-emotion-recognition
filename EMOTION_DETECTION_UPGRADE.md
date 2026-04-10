# Emotion Detection Accuracy Upgrade Guide

## Overview

This upgrade implements a state-of-the-art deep learning approach for emotion detection using Convolutional Neural Networks (CNN) with rich audio features. This significantly improves accuracy compared to the traditional Random Forest model.

## Key Improvements

### 1. Advanced Feature Extraction
- **Mel-Spectrogram**: 2D time-frequency representation (128 mel bands)
- **MFCC**: 40 Mel-Frequency Cepstral Coefficients
- **Chroma Features**: Pitch class information
- **Spectral Contrast**: Frequency band energy differences
- **Zero Crossing Rate**: Signal complexity measure

### 2. Deep Learning Architecture
- **4-layer CNN** with batch normalization and dropout
- **Global Average Pooling** to reduce overfitting
- **Dense layers** with regularization
- **Softmax output** for probability distribution over emotions

### 3. Training Enhancements
- **Data augmentation**: Time stretching, pitch shifting, noise addition
- **Early stopping**: Prevents overfitting
- **Learning rate reduction**: Adaptive learning
- **Stratified split**: Balanced class distribution

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements_deep.txt
```

If you encounter issues with TensorFlow on Windows, use:

```bash
pip install tensorflow-cpu>=2.10.0
```

### Step 2: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## Training the Model

### Step 1: Prepare Data

Ensure your RAVDESS dataset is in the correct location:
```
data/ravdess/
├── Actor_01/
├── Actor_02/
└── ...
```

### Step 2: Train the Deep Learning Model

```bash
python model/train_deep_model.py
```

This will:
- Extract features from all audio files
- Train a CNN model with early stopping
- Save the model to `model/emotion_cnn_model.h5`
- Save the label encoder to `model/label_encoder.pkl`
- Save feature parameters to `model/feature_params.pkl`

Training typically takes 5-15 minutes depending on your hardware.

### Expected Output

```
Starting feature extraction with deep learning approach...
Extracted features from 1440 files.
Feature shape: (1440, 185, 94)
Classes: ['angry' 'calm' 'disgust' 'fearful' 'happy' 'neutral' 'sad' 'surprised']
Number of classes: 8
Training samples: 1152
Test samples: 288

Model architecture:
...

Starting training...
Epoch 1/100
...
Test Accuracy: 0.85+
```

## Using the Model

The application automatically detects and uses the deep learning model if available:

1. **Deep Learning Model** (preferred): Uses `emotion_cnn_model.h5`
2. **Traditional Model** (fallback): Uses `emotion_model.pkl`

### Model Selection Logic

```python
# In app.py
if DEEP_LEARNING_AVAILABLE:
    model = keras.models.load_model("model/emotion_cnn_model.h5")
    model_type = 'deep'
else:
    model = joblib.load("model/emotion_model.pkl")
    model_type = 'traditional'
```

## Expected Accuracy Improvements

| Model Type | Expected Accuracy | Features Used |
|------------|------------------|---------------|
| Traditional (Random Forest) | 60-70% | MFCC only (40 features) |
| Deep Learning (CNN) | 80-90% | Mel-spectrogram + MFCC + Chroma + Contrast + ZCR (185 features) |

## Troubleshooting

### Issue: TensorFlow Installation Fails

**Solution**: Install CPU-only version
```bash
pip uninstall tensorflow
pip install tensorflow-cpu
```

### Issue: Model Training is Slow

**Solution**: Reduce batch size or epochs
```python
# In train_deep_model.py
history = model.fit(
    X_train, y_train_cat,
    epochs=50,  # Reduce from 100
    batch_size=16,  # Reduce from 32
    ...
)
```

### Issue: Out of Memory Error

**Solution**: Process fewer files or reduce feature dimensions
```python
# In train_deep_model.py
mel_spec = librosa.feature.melspectrogram(
    y=y, sr=sr, 
    n_mels=64,  # Reduce from 128
    ...
)
```

### Issue: Model Predicts Same Emotion Always

**Possible Causes**:
1. Imbalanced training data
2. Model overfitting
3. Feature extraction mismatch

**Solution**: Check class distribution and retrain with data augmentation

## Testing the Model

### Test with Sample Audio

```python
from audio_utils.preprocess_deep import extract_features_deep
import tensorflow as tf
import joblib
import numpy as np

# Load model
model = tf.keras.models.load_model("model/emotion_cnn_model.h5")
label_encoder = joblib.load("model/label_encoder.pkl")

# Extract features
features = extract_features_deep("path/to/test_audio.wav")
features_reshaped = np.expand_dims(features, axis=0)

# Predict
predictions = model.predict(features_reshaped)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]
emotion = label_encoder.inverse_transform([predicted_class])[0]

print(f"Predicted Emotion: {emotion}")
print(f"Confidence: {confidence*100:.1f}%")
```

## Model Performance Metrics

After training, you can evaluate detailed metrics:

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print report
print(classification_report(y_test, y_pred_classes, 
                          target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
print(cm)
```

## Next Steps

1. **Train the model**: Run `python model/train_deep_model.py`
2. **Test the application**: Upload audio files and check emotion predictions
3. **Monitor accuracy**: Check confidence scores in predictions
4. **Fine-tune**: Adjust hyperparameters if needed

## Additional Resources

- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [Audio Classification Tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio)

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify RAVDESS dataset is in correct location
3. Ensure audio files are in WAV format
4. Check console output for error messages
