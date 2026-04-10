#!/usr/bin/env python3
"""
Test script to verify speech recognition functionality
"""

import os
import speech_recognition as sr
from audio_utils.record import record_voice

def test_speech_recognition():
    """Test speech recognition with recorded audio"""
    print("Testing speech recognition...")
    
    # Record a short audio clip
    print("Recording 5 seconds of audio...")
    record_voice("test_input.wav", duration=5)
    
    if not os.path.exists("test_input.wav"):
        print("Error: Failed to record audio")
        return False
    
    try:
        # Test speech recognition
        recognizer = sr.Recognizer()
        
        # Configure recognizer for better accuracy
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        
        with sr.AudioFile("test_input.wav") as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            try:
                # Try Google recognition first
                transcription = recognizer.recognize_google(audio_data, language='en-US')
                print(f"✓ Google Speech Recognition: {transcription}")
                return True
            except sr.UnknownValueError:
                print("✗ Google Speech Recognition could not understand audio")
                # Try alternative recognition
                try:
                    transcription = recognizer.recognize_sphinx(audio_data)
                    print(f"✓ Sphinx Speech Recognition: {transcription}")
                    return True
                except Exception as e:
                    print(f"✗ Sphinx Speech Recognition error: {e}")
                    return False
            except sr.RequestError as e:
                print(f"✗ Google Speech Recognition service error: {e}")
                # Try alternative recognition
                try:
                    transcription = recognizer.recognize_sphinx(audio_data)
                    print(f"✓ Sphinx Speech Recognition: {transcription}")
                    return True
                except Exception as e:
                    print(f"✗ Sphinx Speech Recognition error: {e}")
                    return False
        
    except Exception as e:
        print(f"✗ Error during speech recognition testing: {e}")
        return False

if __name__ == "__main__":
    success = test_speech_recognition()
    if success:
        print("\n🎉 Speech recognition is working correctly!")
    else:
        print("\n❌ Speech recognition test failed!")