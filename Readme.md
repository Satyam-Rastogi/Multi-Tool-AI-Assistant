# Multi-Tool Query Assistant

This project consists of a FastAPI backend and a Streamlit frontend that together create a multi-tool query assistant capable of answering questions about physics, weather, stock market, and performing calculations.


## Features

- Physics knowledge base (focused on sound)
- Weather information retrieval
- Stock market data access (If you want to know about a Stock you have to provide the ticker for it as well. Example: Give me information about RELIANCE.NS)
- Basic calculator functionality
- General knowledge queries


## Setup

1. Clone the repository:
```bash
git clone https://github.com/Satyam-Rastogi/Multi-Tool-AI-Assistant.git
```

2. Create a virtual environment (I had used Python 3.11.5 for venv) and install all the dependencies:
```bash
python -m venv multi_tool_assistant
multi_tool_assistant\Scripts\activate
pip install -r requirements.txt
```

3. Make sure to create your own `.env` files in both frontend and backend directories and add your API keys.
    - Create Backend Folder's `.env` file with the following content:
      ```Python 
      GROQ_API_KEY = "Your_Groq_API_Key"
      TOMORROW_IO_API_KEY = "Your_Tomorrow_IO_API_Key"
      ```
    - Create Frontend Folder's `.env` file with the following content:
      ```Python
      SARVAM_API_KEY = "Your_Sarvam_API_Key"
      ```


## Running the Application:

1. Start the FastAPI Backend:
    ```bash
    cd backend
    uvicorn main:app --reload
    ```

2. In a new terminal, start the Streamlit frontend:
    ```bash
    cd frontend
    streamlit run streamlit_app.py
    ```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`)


## Project Demo

Check out this video demonstration of the project:

[![Project Demo Video](https://img.youtube.com/vi/lnQCiRTT0Pk/0.jpg)](https://www.youtube.com/watch?v=lnQCiRTT0Pk)