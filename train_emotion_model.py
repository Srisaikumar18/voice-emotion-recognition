import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

X = np.random.randint(-3000, 3000, size=(100, 1))
y = np.random.choice(
    ["Happy", "Sad", "Angry", "Neutral"], size=100
)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "emotion_model.pkl")

print("Model saved successfully")
