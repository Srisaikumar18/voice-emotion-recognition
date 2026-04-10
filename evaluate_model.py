import os
import numpy as np
import librosa
import joblib

print("Loading model...")
model = joblib.load("model/emotion_model.pkl")

TARGET_SAMPLE_RATE = 16000 
DATA_PATH = "data/ravdess/"
emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def extract_features(file):
    y, sr = librosa.load(file, duration=3, offset=0.5, sr=TARGET_SAMPLE_RATE)
    return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

X, y = [], []
files_to_process = []
for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.wav'):
            files_to_process.append(os.path.join(root, file))

# Limit to 100 files for quick estimation if there are many
np.random.seed(42)
if len(files_to_process) > 200:
    files_to_process = np.random.choice(files_to_process, 200, replace=False)

print(f"Extracting features from {len(files_to_process)} files...")
for filepath in files_to_process:
    try:
        filename = os.path.basename(filepath)
        emotion_code = filename.split("-")[2] 
        if emotion_code in emotions:
            emotion = emotions[emotion_code]
            features = extract_features(filepath)
            X.append(features)
            y.append(emotion)
    except Exception:
        pass

if len(X) > 0:
    accuracy = model.score(X, y)
    print(f"Estimated Accuracy: {accuracy:.2f}")
else:
    print("No data found!")
