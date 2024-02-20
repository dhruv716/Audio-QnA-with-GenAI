from datetime import datetime
import whisper
import sounddevice as sd
import numpy as np
import wave
import tempfile
import csv

def record_and_save_audio():
    """Records audio using the microphone and saves it to a file."""
    
    fs = 44100  # Sampling frequency
    duration = 5  # Recording duration in seconds

    try:
        # Record audio
        audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()

        # Save audio as WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            temp_audio_file.name
            wf = wave.open(temp_audio_file.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio_data.tobytes())
            wf.close()

        return temp_audio_file.name
    except Exception as e:
        print(f"Error saving audio file: {e}")
        return None

def transcribe_audio(file_path):
    try:
        model = whisper.load_model("base")
        # Transcribe audio
        result = model.transcribe(file_path, fp16=False)
        
        # Extract text from the result
        text = result.get("text", "")

        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return "Transcription error"

def save_to_csv(transcript_text, csv_file_path="transcription_output.csv"):
    # Save the transcription to a CSV file
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Transcription"])
        writer.writerow([transcript_text])

    print(f"Transcription saved to {csv_file_path}")
    return csv_file_path  # Return the CSV file path for later use
