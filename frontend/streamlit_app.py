import streamlit as st
import requests
import time
import base64
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Multi-Tool Query Assistant", layout="wide")

st.markdown("<h1 style='text-align: center;'>Multi-Tool Query Assistant</h1>", unsafe_allow_html=True)

st.write("Enter your query:")
st.write("Some examples:")
st.write("- What are the characteristics of Sound?")
st.write("- What's the weather in New York?")
st.write("- Calculate ((5+2)*15)%8")
st.write("- Give me information about RELIANCE.NS")
st.write("--------------")

user_input = st.text_input("Your query:")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

sarvam_api_key = os.getenv("SARVAM_API_KEY")

def text_to_speech(text):
    url = "https://api.sarvam.ai/text-to-speech"
    
    payload = {
        "inputs": [text],
        "target_language_code": "en-IN",
        "speaker": "meera",
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        audio_data = response.json()['audios'][0]

        audio_bytes = base64.b64decode(audio_data)

        return audio_bytes
    except requests.exceptions.HTTPError as http_err:
        error_detail = response.json().get('error', {}).get('message', 'No detailed error message')
        st.error(f"HTTP error occurred: {http_err}. Error details: {error_detail}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"An error occurred while making the request: {req_err}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    return None

def get_query_response(user_input):
    response = requests.post("http://localhost:8000/query", json={"text": user_input})
    if response.status_code == 200:
        data = response.json()
        return data['response']
    else:
        st.error(f"Error in query response: {response.status_code}")
        return None

def process_query(user_input):
    start_time = time.time()
    
    response_text = get_query_response(user_input)
    if not response_text:
        return None, None
    
    elapsed_time = time.time() - start_time
    return response_text, elapsed_time

if 'response_text' not in st.session_state:
    st.session_state.response_text = None

if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = None

if st.button("Get Answer"):
    if user_input:
        with st.spinner("Processing your query..."):
            st.session_state.response_text, st.session_state.elapsed_time = process_query(user_input)
    else:
        st.warning("Please enter a query before submitting.")

if st.session_state.response_text:
    st.markdown(
        f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>"
        f"<strong>Assistant's Response:</strong><br>{st.session_state.response_text}<br><br>"
        f"<span style='font-size: 12px;'><i>Answered in {st.session_state.elapsed_time:.2f} seconds</i></span>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    st.write(" ")
    st.write(" ")

    if st.button("Listen to the Response"):
        with st.spinner("Generating audio..."):
            audio_content = text_to_speech(st.session_state.response_text)
            
            if audio_content:
                audio_file_path = "response_audio.wav"
                with open(audio_file_path, "wb") as f:
                    f.write(audio_content)
                
                st.audio(audio_file_path, format="audio/wav")
                
                with open(audio_file_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Audio",
                        data=file,
                        file_name="response_audio.wav",
                        mime="audio/wav"
                    )
    
    st.write("--------------")

def check_api_status():
    try:
        response = requests.get("http://localhost:8000")
        if response.status_code == 200:
            st.success("API is up and running!")
        else:
            st.warning(f"API returned an unexpected status code: {response.status_code}")
    except requests.RequestException:
        st.error("Unable to connect to the API. Make sure it's running on http://localhost:8000")

if st.button("Check API Status"):
    check_api_status()

st.sidebar.title("Available Tools")
st.sidebar.write("This assistant uses the following tools:")
st.sidebar.write("1. VectorDB: For physics questions about sound from NCERT textbook")
st.sidebar.write("2. WeatherInfo: For current weather and 5-day forecast")
st.sidebar.write("3. Calculator: For mathematical calculations")
st.sidebar.write("4. YFinance: For stock market information")
st.sidebar.write("5. OwnKnowledge: For general knowledge questions")