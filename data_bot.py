# data_bot.py 

import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time

# --- Configuration ---
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Main Application Logic ---
def app():
    st.header("Chat with Data")
    st.markdown("Upload a CSV or Excel file and ask questions about its content.")

    # Initialize session state for chat history and uploaded data
    if "data_messages" not in st.session_state:
        st.session_state.data_messages = [{"role": "assistant", "content": "Please upload a data file to begin."}]
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None

    # --- File Uploader and Data Display ---
    uploaded_file = st.file_uploader("Choose a data file...", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.session_state.uploaded_data = data
            st.dataframe(data, use_container_width=True)
            # Reset chat if a new file is uploaded
            st.session_state.data_messages = [{"role": "assistant", "content": "Data loaded successfully. How can I help you analyze it?"}]
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.uploaded_data = None


    # --- Chat Interface ---
    # Display prior chat messages
    for message in st.session_state.data_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input field at the bottom
    if prompt := st.chat_input("Ask a question about the data..."):
        # Ensure data is uploaded before chatting
        if st.session_state.uploaded_data is None:
            st.warning("Please upload a data file first.")
            st.stop()

        # 1. Add user's prompt to history and display it
        st.session_state.data_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Display a "processing" message and call the model
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # --- Gemini Model Call Logic ---
                default_prompt = """
                You are a data analyst. You will be given a user's question and the full content of a data file. 
                Your job is to analyze the data to answer the user's question. 
                Provide clear, concise answers. Do not return any code.
                """
                
                combined_prompt = f"{default_prompt}\nUser Question: {prompt}"
                data_text = st.session_state.uploaded_data.to_string()

                model = genai.GenerativeModel("gemini-2.5-flash") # Updated to a stable model name
                try:
                    response = model.generate_content([combined_prompt, data_text])
                    response_text = response.text
                except Exception as e:
                    response_text = f"An error occurred: {e}"
                
                st.markdown(response_text)
        
        # 3. Add the model's response to the chat history
        st.session_state.data_messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    app()