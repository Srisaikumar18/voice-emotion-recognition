#!/usr/bin/env python3
"""
Test script to verify audio conversion functionality
"""

import os
import sys
import numpy as np
from scipy.io import wavfile
import librosa

def test_audio_conversion():
    """Test the audio conversion function"""
    
    # Test with a known WAV file from the dataset
    test_file = "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    print(f"Testing audio conversion with: {test_file}")
    
    try:
        # Test the conversion function
        from app import convert_audio_to_wav_simple
        
        # Create a temporary output path
        output_path = "temp_converted.wav"
        
        # Test conversion
        success = convert_audio_to_wav_simple(test_file, output_path)
        
        if success and os.path.exists(output_path):
            print("✓ Audio conversion successful!")
            
            # Verify the converted file
            rate, data = wavfile.read(output_path)
            print(f"✓ Converted file: {rate} Hz, {len(data)} samples")
            
            # Clean up
            os.remove(output_path)
            return True
        else:
            print("✗ Audio conversion failed")
            return False
            
    except Exception as e:
        print(f"✗ Error during conversion test: {e}")
        return False

def test_librosa_loading():
    """Test if librosa can load different audio formats"""
    
    test_file = "data/ravdess/Actor_01/03-01-01-01-01-01-01.wav"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    try:
        print("Testing librosa audio loading...")
        
        # Load audio with librosa
        y, sr = librosa.load(test_file, sr=44100, mono=True)
        
        print(f"✓ Librosa loaded audio: {sr} Hz, {len(y)} samples")
        print(f"✓ Audio range: {y.min():.3f} to {y.max():.3f}")
        
        # Test saving as WAV
        output_path = "temp_librosa_test.wav"
        wavfile.write(output_path, sr, (y * 32767).astype(np.int16))
        
        if os.path.exists(output_path):
            print("✓ Successfully saved as WAV")
            os.remove(output_path)
            return True
        else:
            print("✗ Failed to save as WAV")
            return False
            
    except Exception as e:
        print(f"✗ Error during librosa test: {e}")
        return False

if __name__ == "__main__":
    print("Testing audio conversion functionality...")
    print("=" * 50)
    
    success1 = test_librosa_loading()
    success2 = test_audio_conversion()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Audio conversion is working correctly.")
        print("Your voice recording should now work properly!")
    else:
        print("\n❌ Some tests failed. There may be issues with audio processing.") 