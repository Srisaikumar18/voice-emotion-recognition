"""Test which model is actually being loaded"""
import os
import joblib

print("="*60)
print("MODEL LOADING TEST")
print("="*60)

# Check which model files exist
print("\nChecking model files:")
models = {
    'Improved Model': 'model/emotion_improved_model.pkl',
    'Improved Scaler': 'model/feature_scaler.pkl',
    'Improved Label Encoder': 'model/label_encoder_improved.pkl',
    'Traditional Model': 'model/emotion_model.pkl',
    'Deep Learning Model': 'model/emotion_cnn_model.h5'
}

for name, path in models.items():
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"{exists} {name}: {path}")

# Try loading improved model
print("\n" + "="*60)
print("LOADING IMPROVED MODEL")
print("="*60)

try:
    model = joblib.load("model/emotion_improved_model.pkl")
    scaler = joblib.load("model/feature_scaler.pkl")
    label_encoder = joblib.load("model/label_encoder_improved.pkl")
    print("✅ Improved model loaded successfully!")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Classes: {label_encoder.classes_}")
except Exception as e:
    print(f"❌ Failed to load improved model: {e}")

# Try loading traditional model
print("\n" + "="*60)
print("LOADING TRADITIONAL MODEL")
print("="*60)

try:
    model = joblib.load("model/emotion_model.pkl")
    print("✅ Traditional model loaded successfully!")
    print(f"   Model type: {type(model).__name__}")
except Exception as e:
    print(f"❌ Failed to load traditional model: {e}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

if os.path.exists("model/emotion_improved_model.pkl"):
    print("\n✅ Use the IMPROVED model for best accuracy!")
    print("   The app should automatically use it.")
else:
    print("\n⚠️  Improved model not found.")
    print("   Train it with: python model/train_improved_model.py")
