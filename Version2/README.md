# 🌐 LangChain-Chroma-Llama Integration

Combine the power of LangChain, Chroma, and Llama for audio transcription, semantic search, and natural language understanding. Leverage Whisper ASR, Chroma Vector Store, and LangChain's Llama2Chat model to provide a comprehensive solution.

## 🚀 Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/langchain-chroma-llama.git
    cd langchain-chroma-llama
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    - Additional Dependencies:
        - [FFmpeg](https://ffmpeg.org/): Install FFmpeg for audio file handling. Download it [here](https://ffmpeg.org/download.html).

3. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

## 📂 File Structure

- **`langchain_chroma_llama.py`:** Main script containing the integration logic.
- **`app.py`:** Streamlit app for user interaction and displaying results.
- **`transcribe.py`:** Script for transcribing audio and saving the results.
- **`knowledge.txt`:** Sample document containing knowledge for Chroma.

## 🛠️ Dependencies

- Python
- Streamlit
- Whisper
- NumPy
- Sounddevice
- FFmpeg (for audio file handling)
- CSV
- Chroma
- Chromadb
- OpenAI
- Transformers
- Sentence-Transformers
- Llama
- HuggingFace Hub

## 💡 How It Works

1. **Audio Transcription:**
    - The `transcribe.py` script records audio using the microphone, transcribes it using the Whisper ASR model, and saves the transcription to a CSV file (`transcription_output.csv`).

2. **Chroma Vector Store:**
    - The `knowledge.txt` file contains sample documents. The `langchain_chroma_llama.py` script loads these documents into the Chroma Vector Store.

3. **Semantic Search:**
    - The Streamlit app (`app.py`) allows users to record and transcribe audio. It then performs a semantic search using the LangChain Llama2Chat model and displays the results alongside the top 5 most similar documents from Chroma Vector Store.

## ▶️ Usage

1. Ensure you have the necessary API keys for Chroma and OpenAI. Update `openai_api_key` in the scripts accordingly.

2. Prepare your transcribed text data in a CSV file (e.g., `transcription_output.csv`).

3. Run the Streamlit app (`app.py`) to initiate the integration process.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
