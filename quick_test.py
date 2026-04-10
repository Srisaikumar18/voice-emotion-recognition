#!/usr/bin/env python3
"""
Quick test to verify audio conversion works
"""

import os
from audio_utils.convert import convert_audio_to_wav, ensure_wav_format, validate_audio_file

def main():
    print("Testing audio conversion...")
    
    # Find a WebM file
    upload_dir = "data/user_uploads"
    webm_file = None
    
    if os.path.exists(upload_dir):
        for file in os.listdir(upload_dir):
            if file.endswith('.webm'):
                webm_file = os.path.join(upload_dir, file)
                break
    
    if not webm_file:
        print("No WebM files found. Please record some audio first.")
        return
    
    print(f"Testing with: {webm_file}")
    
    # Test conversion
    wav_path = ensure_wav_format(webm_file)
    
    if wav_path and os.path.exists(wav_path):
        print(f"✓ Conversion successful: {wav_path}")
        
        if validate_audio_file(wav_path):
            print("✓ Audio file is valid!")
            print("🎉 Audio conversion is working correctly!")
        else:
            print("✗ Audio file validation failed")
    else:
        print("✗ Conversion failed")

if __name__ == "__main__":
    main() 