# Whisper_Streamlit_Web_APP

# 🎙️ Audio Transcription App

This project is a simple Streamlit app for recording audio, transcribing it using the Whisper ASR model, and displaying the transcription. The project follows a basic model-view architecture with separate files for the main logic (`main.py`) and the Streamlit app (`streamlit.py`).

Additionally, there is a script named `whisper_test.py` that demonstrates how to transcribe already downloaded audio files using the Whisper ASR model. This script processes audio files within a specified folder and generates a CSV file with audio filenames and their corresponding transcriptions.

## 🚀 Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/dhruv716/Speech-to-text.git
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

     **Additional Dependencies:**
    - [FFmpeg](https://ffmpeg.org/): Make sure to have FFmpeg installed for audio file handling. You can download it [here] (https://ffmpeg.org/download.html).
      

3. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

## ✨ Features

- **Record and Save Audio:** Click the "Record and Save Audio" button to record audio using the microphone and save it as a WAV file.

- **Transcription:** Click the "Output" button to transcribe the recorded audio using the Whisper ASR model.

- **CSV Output:** The transcription is saved to a CSV file (`transcription_output.csv`) for easy access and analysis.

## 📂 File Structure

- **`main.py`:** Contains the main logic for audio recording, transcription, and CSV saving.

- **`streamlit.py`:** Implements the Streamlit app for user interaction.

- **`whisper_test.py`:** Demonstrates how to transcribe already downloaded audio files using the Whisper ASR model.

## 🛠️ Dependencies

- Python
- Streamlit
- Whisper
- NumPy
- Sounddevice
- FFmpeg (for audio file handling)
- CSV 

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.

 