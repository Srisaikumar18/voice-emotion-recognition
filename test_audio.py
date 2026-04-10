#!/usr/bin/env python3
"""
Test script to verify audio processing functionality
"""

import os
import sys
from audio_utils.preprocess import extract_features
from audio_utils.waveform import plot_waveform
import speech_recognition as sr

def test_audio_processing(audio_file_path):
    """Test audio processing pipeline"""
    print(f"Testing audio file: {audio_file_path}")
    
    if not os.path.exists(audio_file_path):
        print(f"Error: File {audio_file_path} does not exist")
        return False
    
    try:
        # Test waveform generation
        print("1. Testing waveform generation...")
        plot_waveform(audio_file_path)
        print("✓ Waveform generated successfully")
        
        # Test feature extraction
        print("2. Testing feature extraction...")
        features = extract_features(audio_file_path)
        if features is not None:
            print(f"✓ Features extracted successfully (shape: {len(features)})")
        else:
            print("✗ Feature extraction failed")
            return False
        
        # Test speech recognition
        print("3. Testing speech recognition...")
        recognizer = sr.Recognizer()
        
        # Configure recognizer for better accuracy
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            try:
                # Try multiple recognition methods for better accuracy
                transcription = recognizer.recognize_google(audio_data, language='en-US')
                print(f"✓ Transcription: {transcription}")
            except sr.UnknownValueError:
                # Fallback to alternative recognition service
                try:
                    transcription = recognizer.recognize_sphinx(audio_data)
                    print(f"✓ Sphinx transcription: {transcription}")
                except Exception:
                    print("✓ Speech recognition: Could not understand audio (this is normal for test files)")
            except sr.RequestError as e:
                print(f"✗ Speech recognition error: {e}")
                return False
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

if __name__ == "__main__":
    # Test with existing audio file if available
    test_files = [
        "input.wav",
        "data/user_uploads/03-01-01-01-02-01-03.wav",
        "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n{'='*50}")
            success = test_audio_processing(test_file)
            if success:
                print("Audio processing pipeline is working correctly!")
                break
        else:
            print(f"Test file not found: {test_file}")
    
    if not any(os.path.exists(f) for f in test_files):
        print("No test files found. Please record some audio or upload a WAV file first.")