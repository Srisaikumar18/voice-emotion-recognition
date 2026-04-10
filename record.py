import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os

def record_voice(filename="input.wav", duration=5, fs=44100):
    """
    Records audio from the microphone and saves it as a WAV file.

    Parameters:
        filename (str): Output filename (default is 'input.wav')
        duration (int): Recording duration in seconds
        fs (int): Sampling rate (default is 44100)
    
    Returns:
        bool: True if recording was successful, False otherwise\
    """
    print("Recording started...")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        write(filename, fs, recording)
        print(f"Recording saved as {filename}")
        
        # Verify the file was created and has content
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            return True
        else:
            print("Recording file was not created or is empty")
            return False
    except Exception as e:
        print(f"Error recording audio: {e}")
        return False
