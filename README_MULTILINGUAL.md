# Multilingual Support for Voice Emotion Tracker

This document describes the multilingual features added to the Voice Emotion Tracker application, with specific support for Tamil (Tamizh) and Telugu languages.

## Features Added

### 1. Multilingual Speech Recognition
- The application now supports multiple Indian languages including Tamil, Telugu, and English
- Users can select their preferred language for transcription from the UI
- Uses Google's Speech Recognition API with appropriate language codes
- Automatically detects and transcribes speech with high accuracy

### 2. Language Selection
- Users can choose the language for transcription from a dropdown menu
- Currently supports Tamil (ta-IN), Telugu (te-IN), English (en-US), Hindi (hi-IN), Kannada (kn-IN), and Malayalam (ml-IN)
- Default language is Tamil for backward compatibility

### 3. Language Detection and Display
- Shows which language was detected for each transcription
- Displays language information in both the main interface and history

### 4. Translation Capabilities
- Provides option to translate Tamil and Telugu transcriptions to English
- Uses Google Translate API for accurate translations
- Simple UI button to trigger translations

### 5. Enhanced Emotion Detection
- Emotion detection works with any language since it's based on audio features, not text
- Speech in any supported language will be processed for emotion detection

## How It Works

### Speech Recognition Process
1. When audio is recorded or uploaded, the system uses the user-selected language for transcription
2. If recognition fails in the selected language, it automatically falls back to other supported languages
3. The detected language is stored with the transcription in the history

### Emotion Detection
- Emotion detection is language-independent since it analyzes audio features (MFCCs) rather than text
- Works equally well with all supported languages

### Translation Feature
- When Tamil or Telugu is detected, a "Translate to English" button appears
- Clicking this button sends the text to our translation API
- The English translation is then displayed below the original transcription

## Supported Languages

Currently, the application supports:
1. **Tamil (ta-IN)** - Tamil language support
2. **Telugu (te-IN)** - Added Telugu language support
3. **English (en-US)** - Fallback language
4. **Hindi (hi-IN)** - Hindi language support
5. **Kannada (kn-IN)** - Kannada language support
6. **Malayalam (ml-IN)** - Malayalam language support

The system can potentially recognize other languages supported by Google Speech Recognition.

## Technical Implementation

### Files Modified
- `app.py` - Added multilingual speech recognition logic and language selection
- `audio_utils/preprocess.py` - Updated for consistency with training
- `app/templates/index.html` - Added language selection dropdown and display
- `app/templates/settings.html` - Added language preference settings
- `app/templates/history.html` - Added language column to history table

### New Files Added
- `audio_utils/translate.py` - Translation utilities with support for multiple languages
- `test_tamil_support.py` - Test script for Tamil recognition
- `test_telugu_support.py` - Test script for Telugu recognition

### Dependencies
The following packages are required for multilingual support:
- `speechrecognition` - For speech-to-text conversion
- `googletrans==4.0.0-rc1` - For translation services

## Testing Language Support

To test the language support:

1. Run the main application:
   ```bash
   python app.py
   ```

2. Open your browser to http://localhost:5000

3. Select your preferred language from the dropdown menu

4. Record or upload audio in that language

5. Verify that the transcription appears in the correct language

6. Check that the language is correctly displayed in the UI

## Testing with Specific Languages

### Tamil Support
- Speak in Tamil when recording audio
- Check that the transcription appears in Tamil
- Verify that "Tamil (ta-IN)" is shown as the detected language

### Telugu Support
- Speak in Telugu when recording audio
- Check that the transcription appears in Telugu
- Verify that "Telugu (te-IN)" is shown as the detected language

## Limitations

1. **Internet Required**: Both speech recognition and translation require internet connectivity
2. **Audio Quality**: Recognition accuracy depends on audio quality
3. **Language Mixing**: Mixed language speech may not be fully recognized

## Future Enhancements

Possible future improvements:
1. Support for additional Indian languages
2. Offline recognition capabilities
3. Improved translation accuracy
4. Language-specific emotion analysis models
5. User preference saving for default language

## Troubleshooting

If language recognition is not working:
1. Ensure you have a stable internet connection
2. Check that your microphone is working properly
3. Speak clearly in the selected language
4. Verify that the audio quality is good

If translation is not working:
1. Check internet connectivity
2. Ensure the googletrans package is properly installed
3. Verify that the translation API endpoint is accessible