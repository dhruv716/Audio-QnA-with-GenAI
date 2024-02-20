# app.py

import streamlit as st
from main import record_and_save_audio, transcribe_audio, save_to_csv

# Function to get or create the session state
def get_session_state():
    if "file_path" not in st.session_state:
        st.session_state.file_path = None
    return st.session_state

def main():
    session_state = get_session_state()

    st.title("Audio Transcription App")
    st.subheader("YOUR_HEADER")

    # Record and Save Audio Button
    if st.button("Record and Save Audio"):
        session_state.file_path = record_and_save_audio()
        if session_state.file_path:
            st.success("Audio recorded successfully.")
        else:
            st.error("Error recording audio.")

    # Output Button
    if st.button("Output"):
        if session_state.file_path:
            transcript_text = transcribe_audio(session_state.file_path)
            
            if transcript_text:
                st.header("Transcription Output:")
                st.write(transcript_text)

                # Save the transcription to a CSV file
                save_to_csv(transcript_text)
            else:
                st.warning("Transcription error.")
        else:
            st.warning("Record audio first using the 'Record and Save Audio' button.")

if __name__ == "__main__":
    main()
