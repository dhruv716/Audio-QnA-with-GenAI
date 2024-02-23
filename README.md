# 🌐 Speech-to-text and semantic search

Combine the power of LangChain, Chroma, and Llama for audio transcription, semantic search, and natural language understanding. Leverage Whisper ASR, Chroma Vector Store, LangChain's Llama2Chat model, and extend the capabilities using Falcon and Faiss for enhanced retrieval and question answering.

## Version 1 - Whisper Streamlit Web App

### 🎙️ Audio Transcription App

This version is a simple Streamlit app for recording audio, transcribing it using the Whisper ASR model, and displaying the transcription. The project follows a basic model-view architecture with separate files for the main logic (`main.py`) and the Streamlit app (`streamlit.py`).

#### 🚀 Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/dhruv716/Speech-to-text.git
    cd Speech-to-text
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

#### ✨ Features

- **Record and Save Audio:** Click the "Record and Save Audio" button to record audio using the microphone and save it as a WAV file.
- **Transcription:** Click the "Output" button to transcribe the recorded audio using the Whisper ASR model.
- **CSV Output:** The transcription is saved to a CSV file (`transcription_output.csv`) for easy access and analysis.

#### 📂 File Structure

- **`main.py`:** Contains the main logic for audio recording, transcription, and CSV saving.
- **`streamlit.py`:** Implements the Streamlit app for user interaction.
- **`whisper_test.py`:** Demonstrates how to transcribe already downloaded audio files using the Whisper ASR model.

#### 🛠️ Dependencies

- Python
- Streamlit
- Whisper
- NumPy
- Sounddevice
- FFmpeg (for audio file handling)
- CSV

#### 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.

---

## Version 2 - LangChain-Chroma-Llama Integration

This version extends the capabilities of the Whisper Streamlit Web App by integrating LangChain, Chroma, Llama, Falcon, and Faiss. It provides a comprehensive solution for audio transcription, semantic search, and enhanced retrieval.

### 🚀 Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/langchain-chroma-llama.git
    cd langchain-chroma-llama/version2
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

### 📂 File Structure

- **`langchain_chroma_llama.py`:** Main script containing the integration logic.
- **`app.py`:** Streamlit app for user interaction and displaying results.
- **`transcribe.py`:** Script for transcribing audio and saving the results.
- **`knowledge.txt`:** Sample document containing knowledge for Chroma.
- **`langchain_faiss_falcon.py`:** Script incorporating Falcon and Faiss for enhanced retrieval and question answering.

### 🛠️ Dependencies

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
- Falcon
- Faiss

### 💡 How It Works

1. **Audio Transcription:**
    - The `transcribe.py` script records audio using the microphone, transcribes it using the Whisper ASR model, and saves the transcription to a CSV file (`transcription_output.csv`).

2. **Chroma Vector Store:**
    - The `knowledge.txt` file contains sample documents. The `langchain_chroma_llama.py` script loads these documents into the Chroma Vector Store.

3. **Semantic Search:**
    - The Streamlit app (`app.py`) allows users to record and transcribe audio. It then performs a semantic search using the LangChain Llama2Chat model and displays the results alongside the top 5 most similar documents from Chroma Vector Store.

4. **Falcon and Faiss Integration:**
    - The `langchain_faiss_falcon.py` script demonstrates the integration of Falcon and Faiss for enhanced retrieval and question answering. It uses the LangChain framework to create a vector store and retrieve relevant information based on a query.

### ▶️ Usage

1. Ensure you have the necessary API keys for Chroma, OpenAI, and HuggingFace Hub. Update the corresponding API keys in the scripts accordingly.

2. Prepare your transcribed text data in a CSV file (e.g., `transcription_output.csv`).

3. Run the Streamlit app (`app.py`) to initiate the integration process.

4. For Falcon and Faiss integration, run `langchain_faiss_falcon.py` to see enhanced retrieval and question answering capabilities.

### 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.

### 📄 License

This project is licensed under the [MIT License](LICENSE).
