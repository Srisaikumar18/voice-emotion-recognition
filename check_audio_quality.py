#!/usr/bin/env python3
"""
Check audio quality for speech recognition
"""

import wave
import numpy as np
import os

def check_audio_quality(file_path):
    """Check if audio file has sufficient quality for speech recognition"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    try:
        with wave.open(file_path, 'rb') as wf:
            # Get audio properties
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            sample_width = wf.getsampwidth()
            
            print(f"Audio Properties:")
            print(f"  Channels: {channels}")
            print(f"  Sample Rate: {sample_rate} Hz")
            print(f"  Duration: {frames/sample_rate:.2f} seconds")
            print(f"  Sample Width: {sample_width} bytes")
            
            # Read audio data
            audio_data = wf.readframes(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate audio statistics
            max_amplitude = np.max(np.abs(audio_array))
            rms = np.sqrt(np.mean(audio_array**2))
            
            print(f"  Max Amplitude: {max_amplitude}")
            print(f"  RMS Amplitude: {rms:.2f}")
            
            # Check quality criteria
            quality_issues = []
            
            # Check if audio is too quiet
            if rms < 100:
                quality_issues.append("Audio is too quiet")
            
            # Check if audio is clipped
            if max_amplitude >= 32700:
                quality_issues.append("Audio may be clipped")
            
            # Check sample rate (should be at least 16000 for good speech recognition)
            if sample_rate < 16000:
                quality_issues.append("Sample rate is too low for good speech recognition")
            
            # Check duration (should be at least 1 second for meaningful recognition)
            if frames/sample_rate < 1.0:
                quality_issues.append("Audio is too short")
            
            if quality_issues:
                print("\nQuality Issues:")
                for issue in quality_issues:
                    print(f"  - {issue}")
                return False
            else:
                print("\nAudio quality is suitable for speech recognition!")
                return True
                
    except Exception as e:
        print(f"Error checking audio quality: {e}")
        return False

if __name__ == "__main__":
    # Check the most recent recording
    test_files = ["input.wav", "test_input.wav"]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"Checking quality of {file_path}:")
            check_audio_quality(file_path)
            break
    else:
        print("No audio files found to check.")