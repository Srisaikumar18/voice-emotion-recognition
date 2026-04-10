#!/usr/bin/env python3
"""
Test script with actual speech for speech recognition
"""

import os
import speech_recognition as sr
from audio_utils.record import record_voice
from audio_utils.enhance import check_and_enhance_audio

def test_with_actual_speech():
    """Test speech recognition with actual speech recording"""
    print("Testing speech recognition with actual speech...")
    print("\nPlease speak clearly into your microphone when prompted.")
    input("Press Enter to start recording (5 seconds)...")
    
    # Record a short audio clip with actual speech
    print("\nRecording 5 seconds of audio (speak now!)...")
    record_voice("actual_speech_test.wav", duration=5)
    
    if not os.path.exists("actual_speech_test.wav"):
        print("❌ Error: Failed to record audio")
        return False
    
    # Enhance the audio
    print("\nEnhancing audio for better recognition...")
    try:
        enhanced_file = check_and_enhance_audio("actual_speech_test.wav")
        print(f"✓ Audio enhanced: {enhanced_file}")
    except Exception as e:
        print(f"❌ Audio enhancement failed: {e}")
        return False
    
    # Test speech recognition with improved settings
    print("\nTesting speech recognition...")
    recognizer = sr.Recognizer()
    
    # Configure recognizer for better accuracy
    recognizer.energy_threshold = 4000  # Increased threshold
    recognizer.dynamic_energy_threshold = False  # Disable dynamic threshold
    recognizer.pause_threshold = 0.8
    
    try:
        with sr.AudioFile(enhanced_file) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)
            
            # Try speech recognition with multiple approaches
            transcription = None
            
            # First try Google Speech Recognition
            try:
                transcription = recognizer.recognize_google(audio_data, language='en-US')
                print(f"✓ Google Speech Recognition: {transcription}")
                return True
            except sr.UnknownValueError:
                print("❌ Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"❌ Google Speech Recognition service error: {e}")
            
            # Fallback to Sphinx
            if transcription is None:
                try:
                    transcription = recognizer.recognize_sphinx(audio_data)
                    print(f"✓ Sphinx Speech Recognition: {transcription}")
                    return True
                except Exception as e:
                    print(f"❌ Sphinx Speech Recognition error: {e}")
            
            # If all methods fail
            if transcription is None:
                print("❌ All speech recognition attempts failed")
                return False
                    
    except Exception as e:
        print(f"❌ Error during speech recognition: {e}")
        return False

if __name__ == "__main__":
    print("Running speech recognition test with actual speech...")
    success = test_with_actual_speech()
    if success:
        print("\n🎉 Speech recognition is working with actual speech!")
    else:
        print("\n❌ Speech recognition is still not working properly.")
        print("\nTroubleshooting tips:")
        print("1. Check your microphone settings in Windows")
        print("2. Ensure you're in a quiet environment")
        print("3. Speak clearly and at a normal volume")
        print("4. Try a different microphone if available")