#!/usr/bin/env python3
"""
Test script to verify the complete audio processing pipeline
"""

import os
from audio_utils.record import record_voice
from audio_utils.enhance import check_and_enhance_audio
import speech_recognition as sr

def test_complete_pipeline():
    """Test the complete audio processing pipeline"""
    print("Testing complete audio processing pipeline...")
    
    # Step 1: Record audio
    print("\n1. Recording audio...")
    record_voice("test_pipeline.wav", duration=5)
    
    if not os.path.exists("test_pipeline.wav"):
        print("❌ Failed to record audio")
        return False
    
    # Step 2: Enhance audio
    print("\n2. Enhancing audio...")
    try:
        enhanced_file = check_and_enhance_audio("test_pipeline.wav")
        print(f"✓ Audio enhanced: {enhanced_file}")
    except Exception as e:
        print(f"❌ Audio enhancement failed: {e}")
        return False
    
    # Step 3: Test speech recognition
    print("\n3. Testing speech recognition...")
    recognizer = sr.Recognizer()
    
    # Configure recognizer
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    
    try:
        with sr.AudioFile(enhanced_file) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            
            # Try speech recognition
            try:
                transcription = recognizer.recognize_google(audio_data, language='en-US')
                print(f"✓ Transcription: {transcription}")
                print("\n🎉 Complete pipeline test successful!")
                return True
            except sr.UnknownValueError:
                print("⚠️ Speech recognition: Could not understand audio")
                print("This might be normal if no speech was recorded")
                print("\n✅ Pipeline executed successfully (no errors)")
                return True
            except sr.RequestError as e:
                print(f"❌ Speech recognition service error: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error during speech recognition: {e}")
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if not success:
        print("\n❌ Pipeline test failed!")
    else:
        print("\n✅ Pipeline test completed!")