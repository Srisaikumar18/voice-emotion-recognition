from flask import Flask, render_template, request, send_file, send_from_directory, redirect, url_for
import os
import joblib
from audio_utils.record import record_voice
from audio_utils.preprocess import extract_features, enhance_audio_for_transcription
try:
    from audio_utils.preprocess_improved import extract_features_improved
    IMPROVED_MODEL_AVAILABLE = True
except ImportError:
    IMPROVED_MODEL_AVAILABLE = False
from audio_utils.waveform import plot_waveform
from audio_utils.convert import ensure_wav_format, validate_audio_file
from audio_utils.translate import translate_to_english, get_supported_languages
import speech_recognition as sr
import json
from datetime import datetime
from scipy.io import wavfile
import shutil
import uuid
import numpy as np

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
# Try to import deep learning dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    from audio_utils.preprocess_deep import extract_features_deep
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("TensorFlow not available. Using traditional model.")

app = Flask(__name__, template_folder="app/templates")

model = None
scaler = None
label_encoder = None
model_type = None  # 'improved', 'deep', or 'traditional'
model_load_error = None

# Try to load improved model first (best accuracy without TensorFlow)
try:
    model = joblib.load(os.path.join("model", "emotion_improved_model.pkl"))
    scaler = joblib.load(os.path.join("model", "feature_scaler.pkl"))
    label_encoder = joblib.load(os.path.join("model", "label_encoder_improved.pkl"))
    model_type = 'improved'
    print("Improved model loaded successfully!")
except Exception as e:
    print(f"Improved model not found: {e}")

# Try deep learning model if improved not available
if model is None and DEEP_LEARNING_AVAILABLE:
    try:
        model = keras.models.load_model(os.path.join("model", "emotion_cnn_model.h5"))
        label_encoder = joblib.load(os.path.join("model", "label_encoder.pkl"))
        model_type = 'deep'
        print("Deep learning model loaded successfully!")
    except Exception as e:
        print(f"Deep learning model not found: {e}")

# Fallback to traditional model
if model is None:
    try:
        model = joblib.load(os.path.join("model", "emotion_model.pkl"))
        model_type = 'traditional'
        print("Traditional model loaded successfully!")
    except Exception as e:
        model_load_error = f"Model loading error: {str(e)}. Please train a model first."
        print(f"Model loading error: {model_load_error}")

HISTORY_FILE = "history.json"
AUDIO_DIR = os.path.join("static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

UPLOAD_FOLDER = os.path.join('data', 'user_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PROCESSING_WAV_PATH = "input.wav" 

SUPPORTED_LANGUAGES = {
    'ta-IN': 'Tamil (ta-IN)',
    'te-IN': 'Telugu (te-IN)',
    'en-US': 'English (en-US)',
    'hi-IN': 'Hindi (hi-IN)',
    'kn-IN': 'Kannada (kn-IN)',
    'ml-IN': 'Malayalam (ml-IN)'
}

def get_audio_properties(filepath):
    """Retrieves audio properties for display."""
    try:
        rate, data = wavfile.read(filepath)
        duration = data.shape[0] / rate
        channels = 1 if len(data.shape) == 1 else data.shape[1]
        return {
            'sample_rate': rate,
            'channels': channels,
            'duration': round(duration, 2),
            'format': 'WAV'
        }
    except Exception:
        return None

@app.route("/", methods=["GET", "POST"])
@app.route("/upload", methods=["GET", "POST"])
def index():
    emotion = None
    features = None
    transcription = None
    detected_language = None  
    audio_available = False
    audio_props = None
    audio_filename = None
    
    if model_load_error:
        emotion = f"Model loading error: {model_load_error}"
    
    if request.method == "POST":
        unique_id = uuid.uuid4().hex
        audio_filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}.wav"
        audio_path_final = os.path.join(AUDIO_DIR, audio_filename)
        
        # Get selected language from form
        selected_language = request.form.get('language', 'ta-IN')  # Default to Tamil
        
        # --- 1. Handle Recording or Upload ---
        if "record" in request.form:
            # The record_voice function saves directly to PROCESSING_WAV_PATH ("input.wav")
            record_voice(PROCESSING_WAV_PATH)
            
        elif "file" in request.files:
            f = request.files["file"]
            if f and f.filename:
                # Save uploaded file temporarily
                save_path = os.path.join(UPLOAD_FOLDER, f.filename)
                f.save(save_path)
                
                # Convert the uploaded file to the standardized WAV path
                wav_path = ensure_wav_format(save_path)
                
                if not wav_path or not validate_audio_file(wav_path):
                    emotion = "Error: Audio file could not be processed. Please ensure it's a valid audio file."
                    if os.path.exists(save_path): os.remove(save_path)
                
        # --- 2. Process Audio (Prediction and Transcription) ---
        if os.path.exists(PROCESSING_WAV_PATH):
            audio_available = True
            
            # --- Emotion Prediction ---
            try:
                # Extract features based on model type
                if model_type == 'improved' and IMPROVED_MODEL_AVAILABLE:
                    features = extract_features_improved(PROCESSING_WAV_PATH)
                    if features is not None and model is not None and scaler is not None:
                        # Scale features
                        features_scaled = scaler.transform([features])
                        
                        # Predict
                        prediction = model.predict(features_scaled)
                        
                        # Get probability if available
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(features_scaled)[0]
                            confidence = np.max(probabilities)
                            predicted_class = np.argmax(probabilities)
                            emotion_label = label_encoder.inverse_transform([predicted_class])[0]
                            emotion = f"{emotion_label} ({confidence*100:.1f}%)"
                        else:
                            emotion = label_encoder.inverse_transform(prediction)[0]
                    else:
                        emotion = "Prediction failed: Feature extraction error."
                        
                elif model_type == 'deep' and DEEP_LEARNING_AVAILABLE:
                    features = extract_features_deep(PROCESSING_WAV_PATH)
                    if features is not None and model is not None:
                        # Reshape for CNN input
                        features_reshaped = np.expand_dims(features, axis=0)
                        
                        # Predict
                        predictions = model.predict(features_reshaped, verbose=0)
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class]
                        
                        # Decode label
                        emotion = label_encoder.inverse_transform([predicted_class])[0]
                        emotion = f"{emotion} ({confidence*100:.1f}%)"
                    else:
                        emotion = "Prediction failed: Feature extraction error."
                else:
                    # Traditional model
                    features = extract_features(PROCESSING_WAV_PATH)
                    if features is not None and model is not None:
                        if isinstance(features, (list, tuple)) or (hasattr(features, 'ndim') and features.ndim > 0 and features.size > 0):
                            prediction = model.predict([features])
                            emotion = prediction[0]
                        else:
                            emotion = "Prediction failed: Feature vector is empty or malformed."
                    elif model is None:
                        emotion = "Model not loaded. Check model files."
                
                # Generate waveform image (for display)
                plot_waveform(PROCESSING_WAV_PATH)
                
            except Exception as e:
                emotion = f"Prediction failed: {e}"
                features = None

            # --- Speech-to-Text Transcription (CRITICAL SECTION) ---
            audio_props = get_audio_properties(PROCESSING_WAV_PATH)
            
            # Enhance audio for better transcription
            enhanced_path = "input_enhanced.wav"
            enhancement_success = False
            
            print(f"[DEBUG] Starting audio enhancement for: {PROCESSING_WAV_PATH}")
            
            try:
                enhancement_success = enhance_audio_for_transcription(PROCESSING_WAV_PATH, enhanced_path)
                if not enhancement_success:
                    print("[DEBUG] Audio enhancement failed, falling back to original audio")
                else:
                    print(f"[DEBUG] Audio enhancement succeeded, using: {enhanced_path}")
            except Exception as e:
                print(f"[DEBUG] Audio enhancement error: {e}, falling back to original audio")
                enhancement_success = False
            
            # Use enhanced audio if available, otherwise fall back to original
            audio_file_for_stt = enhanced_path if enhancement_success else PROCESSING_WAV_PATH
            
            print(f"[DEBUG] Using audio file for STT: {audio_file_for_stt}")
            
            # Initialize Recognizer with robustness settings
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 400  # Lowered for quieter speech
            recognizer.pause_threshold = 0.8  # Increased tolerance for pauses
            
            if audio_props and audio_props['duration'] > 0:
                try:
                    with sr.AudioFile(audio_file_for_stt) as source:
                        # Explicitly read the entire duration of the audio file
                        audio_data = recognizer.record(source, duration=audio_props['duration'])
                        
                        transcription = None
                        detected_language = None
                        
                        # Try recognition with selected language first
                        try:
                            # Using recognize_google for multilingual support
                            # linter issue: reportAttributeAccessIssue - method exists but linter doesn't recognize it
                            transcription = recognizer.recognize_google(audio_data, language=selected_language)  # type: ignore
                            detected_language = SUPPORTED_LANGUAGES.get(selected_language, selected_language)
                        except sr.UnknownValueError:
                            # If selected language recognition fails, try other languages as fallback
                            fallback_languages = ['en-US', 'ta-IN', 'te-IN']
                            for lang in fallback_languages:
                                if lang != selected_language:  # Skip the already tried language
                                    try:
                                        transcription = recognizer.recognize_google(audio_data, language=lang)  # type: ignore
                                        detected_language = SUPPORTED_LANGUAGES.get(lang, lang)
                                        break  # Exit loop if successful
                                    except sr.UnknownValueError:
                                        continue
                                    except sr.RequestError as e:
                                        print(f"STT Request Error for {lang}: {e}")
                                        continue
                            
                            # If all fallbacks fail
                            if transcription is None:
                                transcription = "Could not understand audio. Try speaking louder or clearer."
                                detected_language = "Unknown"
                        except sr.RequestError as e:
                            # Error related to the Google Speech API service (network, quota, etc.)
                            print(f"STT Request Error: {e}")
                            transcription = f"Speech service error: {e}. Check internet connection or API quota."
                            detected_language = "Service Error"
                        except Exception as e:
                             print(f"STT Unexpected Error: {e}")
                             transcription = f"An unexpected transcription error occurred: {e}"
                             detected_language = "Error"

                except Exception as e:
                    print(f"STT Audio File Read Error: {e}")
                    transcription = f"Error reading audio file for STT: {e}"
                    detected_language = "File Error"
                finally:
                    # Cleanup: Remove enhanced temporary file
                    if enhancement_success and os.path.exists(enhanced_path):
                        try:
                            os.remove(enhanced_path)
                        except Exception as e:
                            print(f"Warning: Could not remove temporary enhanced audio file: {e}")

            else:
                transcription = "Error: Audio file is empty or corrupted."
                detected_language = "File Error"
            
            # --- Finalize and Save History ---
            if features is not None and emotion is not None:
                if transcription is None:
                    transcription = "Transcription failed/empty."
                    
                # Copy the processed file to the final history path
                shutil.copy(PROCESSING_WAV_PATH, audio_path_final)

                record = {
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": audio_filename,
                    "emotion": emotion,
                    "transcription": transcription,
                    "language": detected_language,  # Add language information
                    "selected_language": selected_language  # Store the user-selected language
                }
                
                if os.path.exists(HISTORY_FILE):
                    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                        history = json.load(f)
                else:
                    history = []
                history.insert(0, record)
                with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)


        if transcription is None:
            transcription = "No audio recorded or processed yet."
            
    # GET request response
    return render_template("index.html", emotion=emotion, transcription=transcription, audio_available=audio_available, audio_props=audio_props, active_page="home", audio_filename=audio_filename, detected_language=detected_language, supported_languages=SUPPORTED_LANGUAGES)

@app.route("/audio/<filename>")
def audio(filename):
    return send_from_directory(AUDIO_DIR, filename)

@app.route("/history")
def history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    return render_template("history.html", active_page="history", history=history)

@app.route("/analytics")
def analytics():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    emotion_counts = {}
    for record in history:
        emotion = record.get("emotion", "Unknown")
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    return render_template("analytics.html", active_page="analytics", emotion_counts=emotion_counts)

@app.route("/settings")
def settings():
    return render_template("settings.html", active_page="settings", supported_languages=SUPPORTED_LANGUAGES)

@app.route("/translate", methods=["POST"])
def translate_text():
    """API endpoint for translating text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('target_language', 'en')
        
        if not text:
            return {"error": "No text provided"}, 400
            
        # Import translation function here to avoid import issues
        # Handle potential threading issues with translation by using a subprocess approach
        try:
            from audio_utils.translate import translate_text as translate_func
            translated = translate_func(text, target_language)
            return {"translated_text": translated}
        except Exception as translate_error:
            # Log the error for debugging
            print(f"Translation error: {translate_error}")
            # Return a more user-friendly error message
            return {"error": "Translation service temporarily unavailable. Please try again."}, 500
                
    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/delete_history/<int:index>", methods=["POST"])
def delete_history(index):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        if 0 <= index < len(history):
            audio_file = history[index].get("filename")
            if audio_file:
                audio_path = os.path.join(AUDIO_DIR, audio_file)
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            history.pop(index)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
    return redirect(url_for("history"))


if __name__ == "__main__":
    app.run(debug=True)