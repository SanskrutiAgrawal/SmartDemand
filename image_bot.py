# image_bot.py 

import streamlit as st
from PIL import Image
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
    st.header("Chat with Image")
    st.markdown("Upload an image, such as a graph or chart, and ask questions about it.")

    # Initialize session state for chat history and uploaded image
    if "image_messages" not in st.session_state:
        st.session_state.image_messages = [{"role": "assistant", "content": "Please upload an image to begin."}]
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
        
    # --- Image Uploader and Display ---
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Reset chat if a new image is uploaded
        st.session_state.image_messages = [{"role": "assistant", "content": "Image loaded. What would you like to know?"}]

    # --- Chat Interface ---
    # Display prior chat messages
    for message in st.session_state.image_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input field at the bottom
    if prompt := st.chat_input("Ask a question about the image..."):
        # Ensure an image is uploaded before chatting
        if st.session_state.uploaded_image is None:
            st.warning("Please upload an image first.")
            st.stop()

        # 1. Add user's prompt to history and display it
        st.session_state.image_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Display a "processing" message and call the model
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                # --- Gemini Model Call Logic ---
                default_prompt = """
                You are an expert in analyzing forecasting graphs for trend analysis.
                You will receive an input image (which is a graph) and a user's question about it.
                You will have to answer the question based on the observed trends in the image.
                """
                
                combined_prompt = f"{default_prompt}\nUser Question: {prompt}"

                model = genai.GenerativeModel("gemini-2.5-flash") # Updated to a stable model name
                try:
                    # Pass the prompt and the PIL Image object directly
                    response = model.generate_content([combined_prompt, st.session_state.uploaded_image])
                    response_text = response.text
                except Exception as e:
                    response_text = f"An error occurred: {e}"
                
                st.markdown(response_text)
        
        # 3. Add the model's response to the chat history
        st.session_state.image_messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    app()