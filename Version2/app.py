# app.py
import streamlit as st
from transcribe import record_and_save_audio, transcribe_audio, save_to_csv
from LLM_embedding import llama_langchain_chroma_integration, load_chroma_vector_store

# Function to get or create the session state
def get_session_state():
    if "file_path" not in st.session_state:
        st.session_state.file_path = None
    return st.session_state

chroma_vector_store = load_chroma_vector_store()

def main():
    session_state = get_session_state()

    st.title("Audio Transcription and Semantic Search")

    # Record and Transcribe Audio Button
    if st.button("Record and Transcribe Audio"):
        session_state.file_path = record_and_save_audio()
        if session_state.file_path:
            st.success("Audio recorded and transcribed successfully.")
        else:
            st.error("Error recording audio.")

    # Display Transcribed Text
    if session_state.file_path:
        st.header("Transcribed Text:")
        transcribed_text = transcribe_audio(session_state.file_path)
        st.write(transcribed_text)

        # Save the transcription to a CSV file and get the file path
        csv_file_path = save_to_csv(transcribed_text)
        
        openai_api_key = "YOUR_API_KEY"

        # Perform Langchain-Chroma-Llama integration
        results = llama_langchain_chroma_integration(csv_file_path, chroma_vector_store, openai_api_key)
        st.header("Langchain-Chroma-Llama Integration Output:")
        st.write(results)

if __name__ == "__main__":
    main()
