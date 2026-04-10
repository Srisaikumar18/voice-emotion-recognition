import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("🎤 Adjusting for ambient noise...")
    recognizer.adjust_for_ambient_noise(source, duration=1)

    print("🗣️ Speak something now (you have 10 seconds)...")
    try:
        audio = recognizer.listen(source, timeout=10)
        print("✅ Audio captured! Transcribing...")
        text = recognizer.recognize_google(audio)
        print("📝 Transcription:", text)
    except sr.WaitTimeoutError:
        print("⏱️ Timeout: No speech detected.")
    except sr.UnknownValueError:
        print("🤷 Could not understand the audio.")
    except sr.RequestError as e:
        print("🚫 API request error:", e)
