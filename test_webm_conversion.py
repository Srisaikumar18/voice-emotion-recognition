#!/usr/bin/env python3
"""
Test script to verify WebM to WAV conversion functionality
"""

import os
import sys
from audio_utils.convert import convert_audio_to_wav, ensure_wav_format, validate_audio_file

def test_webm_conversion():
    """Test converting a WebM file to WAV"""
    
    # Find a WebM file in the user_uploads directory
    webm_files = []
    upload_dir = "data/user_uploads"
    
    if os.path.exists(upload_dir):
        for file in os.listdir(upload_dir):
            if file.endswith('.webm'):
                webm_files.append(os.path.join(upload_dir, file))
    
    if not webm_files:
        print("No WebM files found in data/user_uploads directory")
        return False
    
    # Test with the first WebM file found
    test_file = webm_files[0]
    print(f"Testing WebM conversion with: {test_file}")
    
    try:
        # Test direct conversion
        output_path = "test_converted.wav"
        success = convert_audio_to_wav(test_file, output_path)
        
        if success and os.path.exists(output_path):
            print("✓ Direct WebM to WAV conversion successful!")
            
            # Test validation
            if validate_audio_file(output_path):
                print("✓ Converted file is valid!")
                
                # Test ensure_wav_format function
                wav_path = ensure_wav_format(test_file)
                if wav_path and os.path.exists(wav_path):
                    print("✓ ensure_wav_format function works!")
                    
                    # Clean up
                    os.remove(output_path)
                    return True
                else:
                    print("✗ ensure_wav_format function failed")
            else:
                print("✗ Converted file is not valid")
        else:
            print("✗ Direct conversion failed")
            
        return False
        
    except Exception as e:
        print(f"✗ Error during conversion test: {e}")
        return False

if __name__ == "__main__":
    print("Testing WebM to WAV conversion...")
    print("=" * 50)
    
    success = test_webm_conversion()
    
    if success:
        print("\n🎉 WebM conversion test passed! Your voice recording should now work properly.")
    else:
        print("\n❌ WebM conversion test failed. There may be issues with audio processing.") 