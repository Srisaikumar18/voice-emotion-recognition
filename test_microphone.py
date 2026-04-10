"""
Microphone diagnostic and test script
"""
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
import os
import time

print("="*60)
print("MICROPHONE DIAGNOSTIC TEST")
print("="*60)

# Step 1: List all audio devices
print("\n1. Available Audio Devices:")
print("-"*60)
devices = sd.query_devices()
print(devices)

print("\n2. Default Input Device:")
print("-"*60)
try:
    default_input = sd.query_devices(kind='input')
    print(default_input)
except Exception as e:
    print(f"Error getting default input: {e}")

# Step 2: Test recording
print("\n3. Testing Microphone Recording:")
print("-"*60)
print("Recording 3 seconds of audio...")
print("Please speak or make noise during recording!")

duration = 3
fs = 16000
test_file = "test_recording.wav"

try:
    # Record
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    
    # Show countdown
    for i in range(duration, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print("Done!")
    
    sd.wait()
    
    # Analyze recording
    max_amplitude = np.max(np.abs(recording))
    mean_amplitude = np.mean(np.abs(recording))
    
    print(f"\n4. Recording Analysis:")
    print("-"*60)
    print(f"Max amplitude: {max_amplitude:.6f}")
    print(f"Mean amplitude: {mean_amplitude:.6f}")
    
    if max_amplitude == 0:
        print("❌ PROBLEM: Recording is completely silent!")
        print("\nPossible causes:")
        print("  1. Microphone is muted or disabled")
        print("  2. Wrong microphone selected")
        print("  3. No microphone permission granted to Python")
        print("  4. Microphone is being used by another application")
        print("\nSolutions:")
        print("  - Check Windows Sound Settings (Right-click speaker icon)")
        print("  - Ensure microphone is not muted")
        print("  - Grant microphone permission to Python")
        print("  - Close other apps using the microphone (Zoom, Teams, etc.)")
    elif max_amplitude < 0.01:
        print("⚠️  WARNING: Recording is very quiet!")
        print(f"   Volume is only {max_amplitude*100:.2f}% of maximum")
        print("\nSuggestions:")
        print("  - Speak louder or move closer to microphone")
        print("  - Increase microphone volume in Windows settings")
        print("  - Check if microphone boost is enabled")
    else:
        print("✅ Recording captured audio successfully!")
        print(f"   Volume level: {max_amplitude*100:.1f}% of maximum")
    
    # Save recording
    recording_clipped = np.clip(recording, -1.0, 1.0)
    recording_int16 = (recording_clipped * 32767).astype(np.int16)
    write(test_file, fs, recording_int16)
    
    print(f"\n5. File Saved:")
    print("-"*60)
    if os.path.exists(test_file):
        file_size = os.path.getsize(test_file)
        print(f"✅ File: {test_file}")
        print(f"   Size: {file_size} bytes")
        
        # Read back and verify
        rate, data = read(test_file)
        print(f"   Sample rate: {rate} Hz")
        print(f"   Duration: {len(data)/rate:.2f} seconds")
        print(f"   Samples: {len(data)}")
    else:
        print("❌ File was not created!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("\nThis might be a permission or driver issue.")

# Step 3: Test with speech recognition
print("\n6. Testing with Speech Recognition:")
print("-"*60)

try:
    import speech_recognition as sr
    
    recognizer = sr.Recognizer()
    
    if os.path.exists(test_file):
        with sr.AudioFile(test_file) as source:
            audio_data = recognizer.record(source)
            
            try:
                text = recognizer.recognize_google(audio_data, language='en-US')
                print(f"✅ Transcription successful: '{text}'")
            except sr.UnknownValueError:
                print("⚠️  Could not understand audio")
                print("   This might be due to:")
                print("   - Audio is too quiet")
                print("   - No speech in recording")
                print("   - Background noise too loud")
            except sr.RequestError as e:
                print(f"❌ API Error: {e}")
    else:
        print("❌ No test file to transcribe")
        
except ImportError:
    print("⚠️  speech_recognition not installed")
except Exception as e:
    print(f"❌ Error: {e}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if os.path.exists(test_file):
    rate, data = read(test_file)
    max_val = np.max(np.abs(data))
    
    if max_val == 0:
        print("❌ MICROPHONE NOT WORKING")
        print("\nNext steps:")
        print("1. Check Windows Sound Settings")
        print("2. Test microphone in Windows Sound Recorder")
        print("3. Grant microphone permissions")
    elif max_val < 1000:  # Very quiet in int16 scale
        print("⚠️  MICROPHONE WORKING BUT VERY QUIET")
        print("\nNext steps:")
        print("1. Increase microphone volume in Windows")
        print("2. Enable microphone boost")
        print("3. Speak louder or move closer")
    else:
        print("✅ MICROPHONE WORKING")
        print("\nIf transcription still fails:")
        print("1. Audio enhancement should help (already implemented)")
        print("2. Try speaking more clearly")
        print("3. Reduce background noise")

print("\nTest file saved as: test_recording.wav")
print("You can play this file to hear what was recorded.")
