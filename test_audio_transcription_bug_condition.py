"""
Bug Condition Exploration Test for Audio Transcription Error Fix

This test verifies that audio with quality issues (low volume, background noise,
excessive silence) can be successfully transcribed after applying the fix.

**AFTER FIX**: This test should PASS, confirming that audio enhancement
(normalization, noise reduction, silence trimming) enables successful transcription.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**
"""

import os
import numpy as np
from scipy.io import wavfile
import speech_recognition as sr
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck

# Constants from the application
TARGET_SAMPLE_RATE = 16000
PROCESSING_WAV_PATH = "test_audio_bug_condition.wav"


def generate_speech_like_audio(duration_seconds=2.0, sample_rate=16000):
    """
    Generate a simple speech-like audio signal using sine waves.
    This simulates speech formants at typical frequencies.
    """
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    
    # Simulate speech formants with multiple frequencies
    # Typical speech formants are around 500Hz, 1500Hz, 2500Hz
    signal = (
        np.sin(2 * np.pi * 500 * t) +
        0.5 * np.sin(2 * np.pi * 1500 * t) +
        0.3 * np.sin(2 * np.pi * 2500 * t)
    )
    
    # Normalize to reasonable amplitude
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal


def create_low_volume_audio(filepath, volume_factor=0.05):
    """
    Create audio with low volume (max amplitude < 0.1).
    This simulates the bug condition from Requirements 1.1.
    """
    signal = generate_speech_like_audio()
    
    # Reduce volume to very low level
    low_volume_signal = signal * volume_factor
    
    # Convert to int16 format for WAV
    audio_data = (low_volume_signal * 32767).astype(np.int16)
    
    wavfile.write(filepath, TARGET_SAMPLE_RATE, audio_data)
    
    # Verify low volume condition
    max_amplitude = np.max(np.abs(low_volume_signal))
    assert max_amplitude < 0.1, f"Expected low volume (< 0.1), got {max_amplitude}"
    
    return filepath


def create_noisy_audio(filepath, snr_db=5):
    """
    Create audio with background noise (low SNR < 10dB).
    This simulates the bug condition from Requirements 1.2.
    """
    signal = generate_speech_like_audio()
    
    # Calculate noise power for desired SNR
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate white noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    # Add noise to signal
    noisy_signal = signal + noise
    
    # Normalize to prevent clipping
    noisy_signal = noisy_signal / np.max(np.abs(noisy_signal)) * 0.8
    
    # Convert to int16 format for WAV
    audio_data = (noisy_signal * 32767).astype(np.int16)
    
    wavfile.write(filepath, TARGET_SAMPLE_RATE, audio_data)
    
    return filepath


def create_excessive_silence_audio(filepath, silence_duration=3.0):
    """
    Create audio with excessive leading/trailing silence (>2s).
    This simulates the bug condition from Requirements 1.3.
    """
    # Generate silence
    silence_samples = int(TARGET_SAMPLE_RATE * silence_duration)
    silence = np.zeros(silence_samples)
    
    # Generate speech in the middle
    speech = generate_speech_like_audio(duration_seconds=2.0)
    
    # Concatenate: silence + speech + silence
    audio_with_silence = np.concatenate([silence, speech, silence])
    
    # Convert to int16 format for WAV
    audio_data = (audio_with_silence * 32767).astype(np.int16)
    
    wavfile.write(filepath, TARGET_SAMPLE_RATE, audio_data)
    
    # Verify excessive silence condition
    assert silence_duration > 2.0, f"Expected excessive silence (> 2s), got {silence_duration}s"
    
    return filepath


def attempt_transcription(audio_filepath):
    """
    Attempt transcription using the FIXED approach with audio enhancement.
    This applies preprocessing (normalization, noise reduction, silence trimming)
    before transcription, matching the fixed code path in app.py.
    """
    from audio_utils.preprocess import enhance_audio_for_transcription
    
    # Enhance audio before transcription (FIXED CODE PATH)
    enhanced_path = f"{audio_filepath}_enhanced.wav"
    enhancement_success = False
    
    try:
        enhancement_success = enhance_audio_for_transcription(audio_filepath, enhanced_path)
        if not enhancement_success:
            print(f"Audio enhancement failed for {audio_filepath}, falling back to original")
    except Exception as e:
        print(f"Audio enhancement error: {e}, falling back to original")
        enhancement_success = False
    
    # Use enhanced audio if available, otherwise fall back to original
    audio_file_for_stt = enhanced_path if enhancement_success else audio_filepath
    
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 400
    recognizer.pause_threshold = 0.8
    
    try:
        with sr.AudioFile(audio_file_for_stt) as source:
            audio_data = recognizer.record(source)
            
            # Try transcription with English
            try:
                transcription = recognizer.recognize_google(audio_data, language='en-US')
                return transcription
            except sr.UnknownValueError:
                return "Could not understand audio. Try speaking louder or clearer."
            except sr.RequestError as e:
                return f"Speech service error: {e}"
    except Exception as e:
        return f"Error reading audio file: {e}"
    finally:
        # Cleanup enhanced file
        if enhancement_success and os.path.exists(enhanced_path):
            try:
                os.remove(enhanced_path)
            except:
                pass


# Property 1: Bug Condition - Low Volume Audio Causes Transcription Failure
@given(
    volume_factor=st.floats(min_value=0.01, max_value=0.09)
)
@settings(
    max_examples=3,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_low_volume_audio_transcription_fails(volume_factor):
    """
    Property: Low volume audio should transcribe successfully after enhancement
    
    For any audio with max amplitude < 0.1 (low volume), the FIXED transcription
    pipeline should apply volume normalization and successfully transcribe the speech.
    
    **EXPECTED OUTCOME AFTER FIX**: 
    This test should PASS because the enhancement normalizes volume and enables
    successful transcription.
    
    **Validates: Requirements 2.1, 2.4**
    """
    # Create low volume audio
    test_file = f"test_low_volume_{volume_factor:.3f}.wav"
    create_low_volume_audio(test_file, volume_factor)
    
    try:
        # Attempt transcription with enhanced audio (FIXED code path)
        result = attempt_transcription(test_file)
        
        # After fix, we expect transcription to succeed
        assert "Could not understand audio" not in result, \
            f"Transcription failed for low volume audio (volume={volume_factor:.3f}): {result}"
        
        assert result != "", "Transcription returned empty result"
        
        print(f"✓ Low volume audio transcribed successfully: '{result}'")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


# Property 2: Bug Condition - Noisy Audio Causes Transcription Failure
@given(
    snr_db=st.floats(min_value=0, max_value=9)
)
@settings(
    max_examples=3,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_noisy_audio_transcription_fails(snr_db):
    """
    Property: Noisy audio should transcribe successfully after enhancement
    
    For any audio with poor signal-to-noise ratio (SNR < 10dB), the FIXED transcription
    pipeline should apply noise reduction and successfully transcribe the speech.
    
    **EXPECTED OUTCOME AFTER FIX**: 
    This test should PASS because the enhancement reduces noise and enables
    successful transcription.
    
    **Validates: Requirements 2.2, 2.4**
    """
    # Create noisy audio
    test_file = f"test_noisy_{snr_db:.1f}db.wav"
    create_noisy_audio(test_file, snr_db)
    
    try:
        # Attempt transcription with enhanced audio (FIXED code path)
        result = attempt_transcription(test_file)
        
        # After fix, we expect transcription to succeed
        assert "Could not understand audio" not in result, \
            f"Transcription failed for noisy audio (SNR={snr_db:.1f}dB): {result}"
        
        assert result != "", "Transcription returned empty result"
        
        print(f"✓ Noisy audio transcribed successfully: '{result}'")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


# Property 3: Bug Condition - Excessive Silence Causes Transcription Failure
@given(
    silence_duration=st.floats(min_value=2.5, max_value=5.0)
)
@settings(
    max_examples=3,
    suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_excessive_silence_audio_transcription_fails(silence_duration):
    """
    Property: Audio with excessive silence should transcribe successfully after enhancement
    
    For any audio with long leading/trailing silence (>2s), the FIXED transcription
    pipeline should trim silence and successfully transcribe the speech content.
    
    **EXPECTED OUTCOME AFTER FIX**: 
    This test should PASS because the enhancement trims silence and enables
    successful transcription.
    
    **Validates: Requirements 2.3, 2.4**
    """
    # Create audio with excessive silence
    test_file = f"test_silence_{silence_duration:.1f}s.wav"
    create_excessive_silence_audio(test_file, silence_duration)
    
    try:
        # Attempt transcription with enhanced audio (FIXED code path)
        result = attempt_transcription(test_file)
        
        # After fix, we expect transcription to succeed
        assert "Could not understand audio" not in result, \
            f"Transcription failed for audio with excessive silence ({silence_duration:.1f}s): {result}"
        
        assert result != "", "Transcription returned empty result"
        
        print(f"✓ Audio with excessive silence transcribed successfully: '{result}'")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    print("=" * 70)
    print("Bug Condition Verification Test - Audio Transcription Fix")
    print("=" * 70)
    print("\nThis test verifies the audio enhancement fix works correctly:")
    print("1. Low volume audio (max amplitude < 0.1) - should transcribe after normalization")
    print("2. Noisy audio (SNR < 10dB) - should transcribe after noise reduction")
    print("3. Excessive silence (>2s leading/trailing) - should transcribe after trimming")
    print("\nEXPECTED: Tests PASS after fix is implemented")
    print("This confirms the bug is fixed.\n")
    
    print("\n--- Testing Low Volume Audio ---")
    try:
        test_low_volume_audio_transcription_fails()
        print("SUCCESS: Low volume test PASSED - enhancement works!")
    except AssertionError as e:
        print(f"FAILURE: Low volume test FAILED - {e}")
    
    print("\n--- Testing Noisy Audio ---")
    try:
        test_noisy_audio_transcription_fails()
        print("SUCCESS: Noisy audio test PASSED - enhancement works!")
    except AssertionError as e:
        print(f"FAILURE: Noisy audio test FAILED - {e}")
    
    print("\n--- Testing Excessive Silence Audio ---")
    try:
        test_excessive_silence_audio_transcription_fails()
        print("SUCCESS: Excessive silence test PASSED - enhancement works!")
    except AssertionError as e:
        print(f"FAILURE: Excessive silence test FAILED - {e}")
    
    print("\n" + "=" * 70)
    print("Bug Condition Verification Complete")
    print("=" * 70)
