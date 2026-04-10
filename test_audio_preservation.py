"""
Preservation Property Tests for Audio Transcription Fix

This test suite validates that high-quality audio behavior remains unchanged
after implementing the audio enhancement fix. These tests run on UNFIXED code
to establish baseline behavior that must be preserved.

**CRITICAL**: These tests are EXPECTED TO PASS on unfixed code.
Passing confirms the baseline behavior we need to preserve.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""

import os
import numpy as np
import librosa
import soundfile as sf
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck
import speech_recognition as sr
from audio_utils.preprocess import extract_features, TARGET_SAMPLE_RATE
import tempfile


# Strategy: Generate high-quality audio samples
@st.composite
def high_quality_audio(draw):
    """
    Generate high-quality audio samples that should transcribe successfully.
    
    Characteristics of high-quality audio:
    - Good volume: amplitude in range [0.3, 0.9]
    - Minimal noise: clean signal
    - Minimal silence: short or no leading/trailing silence
    - Duration: 1-3 seconds (typical for speech)
    """
    duration = draw(st.floats(min_value=1.0, max_value=3.0))
    num_samples = int(duration * TARGET_SAMPLE_RATE)
    
    # Generate a clean sine wave as a proxy for speech
    # Frequency range typical for human speech: 85-255 Hz
    frequency = draw(st.floats(min_value=85.0, max_value=255.0))
    
    # Good amplitude (not too quiet, not clipping)
    amplitude = draw(st.floats(min_value=0.3, max_value=0.9))
    
    # Generate clean audio signal
    t = np.linspace(0, duration, num_samples)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add minimal silence (< 0.5 seconds) at beginning/end
    silence_duration = draw(st.floats(min_value=0.0, max_value=0.5))
    silence_samples = int(silence_duration * TARGET_SAMPLE_RATE)
    
    if silence_samples > 0:
        silence = np.zeros(silence_samples)
        audio = np.concatenate([silence, audio, silence])
    
    return audio.astype(np.float32)


# Property 2.1: High-Quality Audio Sample Rate Preservation
@given(audio=high_quality_audio())
@settings(
    max_examples=3,  # Fast execution as required
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None  # Disable deadline for file I/O operations
)
def test_sample_rate_preservation(audio):
    """
    Property: 16kHz sample rate is maintained throughout pipeline
    
    For any high-quality audio input, the system should:
    1. Maintain 16kHz sample rate when saving to file
    2. Load audio at 16kHz for feature extraction
    3. Preserve 16kHz for STT compatibility
    
    **Validates: Requirement 3.4**
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save audio at 16kHz
        sf.write(tmp_path, audio, TARGET_SAMPLE_RATE)
        
        # Verify file was saved at correct sample rate
        loaded_audio, loaded_sr = librosa.load(tmp_path, sr=None)
        assert loaded_sr == TARGET_SAMPLE_RATE, \
            f"Sample rate mismatch: expected {TARGET_SAMPLE_RATE}, got {loaded_sr}"
        
        # Verify feature extraction uses 16kHz
        features = extract_features(tmp_path)
        assert features is not None, "Feature extraction failed"
        assert len(features) == 40, "MFCC features should have 40 coefficients"
        
        print(f"✓ Sample rate preserved: {TARGET_SAMPLE_RATE} Hz, features shape: {features.shape}")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Property 2.2: Emotion Detection Preservation
@given(audio=high_quality_audio())
@settings(
    max_examples=3,  # Fast execution as required
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None  # Disable deadline for file I/O operations
)
def test_emotion_detection_preservation(audio):
    """
    Property: Emotion detection continues to work correctly
    
    For any high-quality audio input, the system should:
    1. Extract MFCC features successfully
    2. Features should be 40-dimensional
    3. Features should be valid (no NaN, no Inf)
    4. Feature extraction should not fail
    
    **Validates: Requirement 3.5**
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save audio
        sf.write(tmp_path, audio, TARGET_SAMPLE_RATE)
        
        # Extract MFCC features (used for emotion detection)
        features = extract_features(tmp_path)
        
        # Verify features are valid
        assert features is not None, "Feature extraction returned None"
        assert len(features) == 40, f"Expected 40 MFCC coefficients, got {len(features)}"
        assert not np.any(np.isnan(features)), "Features contain NaN values"
        assert not np.any(np.isinf(features)), "Features contain Inf values"
        
        # Verify features are in reasonable range for MFCC
        # MFCC values can vary widely, but should be finite
        assert np.all(np.isfinite(features)), "MFCC features contain non-finite values"
        
        print(f"✓ Emotion detection preserved: 40 MFCC features extracted, range: [{features.min():.2f}, {features.max():.2f}]")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Property 2.3: High-Quality Audio Transcription Accuracy
@given(audio=high_quality_audio())
@settings(
    max_examples=3,  # Fast execution as required
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None  # Disable deadline for API calls
)
def test_high_quality_audio_transcription_preserved(audio):
    """
    Property: High-quality audio transcription accuracy is maintained
    
    For any high-quality audio input (good volume, low noise, minimal silence),
    the system should:
    1. Process the audio without errors
    2. Not degrade transcription quality
    3. Handle audio file operations correctly
    
    Note: We cannot test actual transcription without real speech audio,
    but we can verify the audio processing pipeline works correctly.
    
    **Validates: Requirement 3.1**
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save high-quality audio
        sf.write(tmp_path, audio, TARGET_SAMPLE_RATE)
        
        # Verify audio file is readable by speech_recognition library
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 400
        recognizer.pause_threshold = 0.8
        
        with sr.AudioFile(tmp_path) as source:
            # Verify we can read the audio file
            audio_data = recognizer.record(source)
            assert audio_data is not None, "Failed to read audio data"
            
            # Verify audio properties
            from scipy.io import wavfile
            rate, data = wavfile.read(tmp_path)
            assert rate == TARGET_SAMPLE_RATE, f"Sample rate mismatch: {rate} != {TARGET_SAMPLE_RATE}"
            
            # Verify audio has good volume (not too quiet)
            max_amplitude = np.max(np.abs(data))
            assert max_amplitude > 0.1, f"Audio too quiet: max amplitude {max_amplitude}"
            
            print(f"✓ High-quality audio processing preserved: {rate} Hz, max amplitude: {max_amplitude:.3f}")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Property 2.4: Audio File Format Preservation
@given(audio=high_quality_audio())
@settings(
    max_examples=3,  # Fast execution as required
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None  # Disable deadline for file I/O operations
)
def test_audio_format_preservation(audio):
    """
    Property: WAV format and audio properties are preserved
    
    For any audio input, the system should:
    1. Save audio in WAV format
    2. Preserve audio duration
    3. Maintain mono channel (1 channel)
    4. Keep 16kHz sample rate
    
    **Validates: Requirements 3.1, 3.4**
    **EXPECTED OUTCOME ON UNFIXED CODE**: Test PASSES
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save audio
        sf.write(tmp_path, audio, TARGET_SAMPLE_RATE)
        
        # Verify file exists and is readable
        assert os.path.exists(tmp_path), "Audio file was not created"
        assert os.path.getsize(tmp_path) > 0, "Audio file is empty"
        
        # Verify audio properties
        from scipy.io import wavfile
        rate, data = wavfile.read(tmp_path)
        
        assert rate == TARGET_SAMPLE_RATE, f"Sample rate not preserved: {rate} != {TARGET_SAMPLE_RATE}"
        
        # Verify mono channel
        if len(data.shape) == 1:
            channels = 1
        else:
            channels = data.shape[1]
        assert channels == 1, f"Expected mono audio, got {channels} channels"
        
        # Verify duration is reasonable
        duration = len(data) / rate
        expected_duration = len(audio) / TARGET_SAMPLE_RATE
        assert abs(duration - expected_duration) < 0.1, \
            f"Duration mismatch: {duration:.2f}s != {expected_duration:.2f}s"
        
        print(f"✓ Audio format preserved: WAV, {rate} Hz, {channels} channel(s), {duration:.2f}s")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print("=" * 70)
    print("Preservation Property Tests")
    print("=" * 70)
    print("\nThese tests validate that high-quality audio behavior is preserved.")
    print("EXPECTED: All tests PASS on unfixed code")
    print("This confirms baseline behavior to preserve.\n")
    
    try:
        print("\n--- Test 1: Sample Rate Preservation ---")
        test_sample_rate_preservation()
        
        print("\n--- Test 2: Emotion Detection Preservation ---")
        test_emotion_detection_preservation()
        
        print("\n--- Test 3: High-Quality Audio Transcription Preserved ---")
        test_high_quality_audio_transcription_preserved()
        
        print("\n--- Test 4: Audio Format Preservation ---")
        test_audio_format_preservation()
        
        print("\n" + "=" * 70)
        print("SUCCESS: All Preservation Tests PASSED")
        print("=" * 70)
        print("\nBaseline behavior confirmed:")
        print("✓ 16kHz sample rate maintained throughout pipeline")
        print("✓ Emotion detection (MFCC extraction) works correctly")
        print("✓ High-quality audio processing preserved")
        print("✓ WAV format and audio properties preserved")
        print("\nThese behaviors must remain unchanged after implementing the fix!")
        
    except AssertionError as e:
        print("\n" + "=" * 70)
        print("UNEXPECTED: Test FAILED")
        print("=" * 70)
        print(f"\nAssertion Error: {e}")
        print("\nThis is unexpected - preservation tests should pass on unfixed code.")
        print("Investigation needed to understand why baseline behavior is not as expected.")
        raise
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
        raise
