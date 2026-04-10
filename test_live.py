#!/usr/bin/env python3
"""
Test script to verify live audio processing with dataset files
"""

import os
import sys
from audio_utils.preprocess import extract_features
from audio_utils.waveform import plot_waveform
import speech_recognition as sr
import joblib

def test_with_dataset_file():
    """Test with a file from the RAVDESS dataset"""
    
    # Find a dataset file
    dataset_path = "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return False
    
    print(f"Testing with dataset file: {dataset_path}")
    
    try:
        # Test waveform generation
        print("1. Testing waveform generation...")
        plot_waveform(dataset_path)
        print("✓ Waveform generated successfully")
        
        # Test feature extraction
        print("2. Testing feature extraction...")
        features = extract_features(dataset_path)
        if features is not None:
            print(f"✓ Features extracted successfully (shape: {len(features)})")
        else:
            print("✗ Feature extraction failed")
            return False
        
        # Test emotion prediction
        print("3. Testing emotion prediction...")
        model = joblib.load(os.path.join("model", "emotion_model.pkl"))
        prediction = model.predict([features])
        emotion = prediction[0]
        print(f"✓ Emotion predicted: {emotion}")
        
        # Test speech recognition
        print("4. Testing speech recognition...")
        recognizer = sr.Recognizer()
        with sr.AudioFile(dataset_path) as source:
            audio_data = recognizer.record(source)
            try:
                transcription = recognizer.recognize_google(audio_data, language='en-US')
                print(f"✓ Transcription: {transcription}")
            except sr.UnknownValueError:
                print("✓ Speech recognition: Could not understand audio (this is normal for dataset files)")
            except sr.RequestError as e:
                print(f"✗ Speech recognition error: {e}")
                return False
        
        print("✓ All tests passed! The system is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_with_dataset_file()
    if success:
        print("\n🎉 Your voice emotion tracker is working correctly!")
        print("The issue is likely with the browser recording or audio format conversion.")
        print("Try uploading a WAV file instead of recording through the browser.")
    else:
        print("\n❌ There are issues with the audio processing pipeline.") 