# multipage.py 

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai

# These imports should be at the top
import main
import image_bot
import data_bot

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page configuration - this should be the first Streamlit command
st.set_page_config(
    page_title="Smart Forecaster",
    page_icon="🔮",
    layout="wide" # Use wide layout for a more modern feel
)

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # --- UI IMPROVEMENT: Add a main title for the entire app ---
        st.title("🔮 Smart Forecasting Suite")

        # Initialize session state for the menu
        if 'selected_index' not in st.session_state:
            st.session_state.selected_index = 0
      
        # --- UI IMPROVEMENT: Slightly tweaked menu for a cleaner look ---
        selected = option_menu(
            menu_title=None, # Removed the empty menu_title
            options=['Generate Forecasts','Chat with Image', 'Chat with Data'],
            icons=['graph-up-arrow', 'image', 'chat-left-text-fill'], # More distinct icons
            menu_icon='cast',
            default_index=st.session_state.selected_index,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa", "border-radius": "10px"},
                "icon": {"color": "#333", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#02ab21", "color": "black"},
            }
        )

        st.session_state.selected_index = ['Generate Forecasts', 'Chat with Image', 'Chat with Data'].index(selected)

        # Add a visual separator
        st.markdown("---")
        
        # Page routing
        if selected == "Generate Forecasts":
            main.app()
        elif selected == "Chat with Image":
            image_bot.app()
        elif selected == "Chat with Data":
            data_bot.app()
             
    run()