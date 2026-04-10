"""
Diagnostic script to check audio recording and loading
"""
import numpy as np
import librosa
from scipy.io import wavfile
import soundfile as sf

# Check the input.wav file
print("=" * 70)
print("AUDIO DIAGNOSTIC REPORT")
print("=" * 70)

filename = "input.wav"

try:
    # Method 1: scipy.io.wavfile
    print("\n1. Reading with scipy.io.wavfile:")
    sample_rate_scipy, data_scipy = wavfile.read(filename)
    print(f"   Sample rate: {sample_rate_scipy} Hz")
    print(f"   Data type: {data_scipy.dtype}")
    print(f"   Shape: {data_scipy.shape}")
    print(f"   Min value: {np.min(data_scipy)}")
    print(f"   Max value: {np.max(data_scipy)}")
    print(f"   Mean: {np.mean(data_scipy):.4f}")
    print(f"   Std dev: {np.std(data_scipy):.4f}")
    
    # Method 2: librosa
    print("\n2. Reading with librosa:")
    data_librosa, sample_rate_librosa = librosa.load(filename, sr=None)
    print(f"   Sample rate: {sample_rate_librosa} Hz")
    print(f"   Data type: {data_librosa.dtype}")
    print(f"   Shape: {data_librosa.shape}")
    print(f"   Min value: {np.min(data_librosa):.6f}")
    print(f"   Max value: {np.max(data_librosa):.6f}")
    print(f"   Mean: {np.mean(data_librosa):.6f}")
    print(f"   Std dev: {np.std(data_librosa):.6f}")
    
    # Method 3: soundfile
    print("\n3. Reading with soundfile:")
    data_sf, sample_rate_sf = sf.read(filename)
    print(f"   Sample rate: {sample_rate_sf} Hz")
    print(f"   Data type: {data_sf.dtype}")
    print(f"   Shape: {data_sf.shape}")
    print(f"   Min value: {np.min(data_sf):.6f}")
    print(f"   Max value: {np.max(data_sf):.6f}")
    print(f"   Mean: {np.mean(data_sf):.6f}")
    print(f"   Std dev: {np.std(data_sf):.6f}")
    
    # Check if audio is actually silent
    print("\n4. Audio Content Analysis:")
    if np.max(np.abs(data_scipy)) == 0:
        print("   ❌ PROBLEM: Audio is completely silent (all zeros)")
        print("   This means the microphone is not capturing audio")
    else:
        print(f"   ✓ Audio contains data (max amplitude: {np.max(np.abs(data_scipy))})")
    
    # Check for clipping
    if data_scipy.dtype == np.int16:
        if np.max(data_scipy) >= 32767 or np.min(data_scipy) <= -32768:
            print("   ⚠ WARNING: Audio may be clipping")
    
    # Energy analysis
    energy = np.sum(data_scipy.astype(float) ** 2) / len(data_scipy)
    print(f"   Energy: {energy:.2f}")
    
    if energy < 100:
        print("   ⚠ WARNING: Very low energy - audio might be too quiet")
    
except Exception as e:
    print(f"\n❌ ERROR reading audio file: {e}")

print("\n" + "=" * 70)
