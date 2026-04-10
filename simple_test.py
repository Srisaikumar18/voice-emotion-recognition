#!/usr/bin/env python3
"""
Simple test to verify speech recognition improvements
"""

import speech_recognition as sr
import os

def test_speech_recognition():
    """Test if speech recognition is working"""
    print("Testing speech recognition...")
    
    # Check if we have any audio files to test with
    test_files = ["input.wav", "test_improvements.wav", "actual_speech_test.wav"]
    audio_file = None
    
    for file in test_files:
        if os.path.exists(file):
            audio_file = file
            break
    
    if not audio_file:
        print("No audio files found to test with.")
        return False
    
    print(f"Testing with audio file: {audio_file}")
    
    # Test speech recognition
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold = 0.8
    
    try:
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)
            
            try:
                transcription = recognizer.recognize_google(audio_data, language='en-US')
                print(f"✓ Google Speech Recognition: {transcription}")
                return True
            except sr.UnknownValueError:
                print("⚠️ Google Speech Recognition could not understand audio")
                return True
            except sr.RequestError as e:
                print(f"❌ Google Speech Recognition service error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error during speech recognition: {e}")
        return False

if __name__ == "__main__":
    test_speech_recognition()