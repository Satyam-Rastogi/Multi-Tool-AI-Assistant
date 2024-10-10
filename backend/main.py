import os
import requests
import re
import yfinance as yf
import uvicorn
from dotenv import load_dotenv
from typing import List, Union
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentOutputParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "API is running"}

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

chat_model = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

doc_qa_prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question

    {context}

    Questions:{input}
    """
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
loader = PyPDFLoader("NCERT_Physics_Sound.pdf")
docs = loader.load()

chunker = SemanticChunker(
    embeddings=embeddings,
    buffer_size=1,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=0.8
)

final_documents = []
for doc in docs[:30]:
    chunks = chunker.create_documents([doc.page_content])
    final_documents.extend(chunks)

vector_store = FAISS.from_documents(final_documents, embeddings)

def query_vectordb(query: str) -> str:
    document_chain = create_stuff_documents_chain(chat_model, doc_qa_prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': query})
    
    context = "\n\n".join([doc.page_content for doc in response['context']])
    
    full_response = f"Context:\n{context}\n\nAnswer:\n{response['answer']}"
    return full_response

def extract_location_llm(query: str) -> str:
    prompt = f"""
    Extract the location mentioned in the following query. If multiple locations are mentioned, choose the most likely one that the query is asking about. If no location is mentioned, respond with "No location found".

    Query: {query}

    Location:
    """
    
    response = chat_model.invoke(prompt)
    extracted_location = response.content.strip()
    
    return None if extracted_location == "No location found" else extracted_location

def get_weather(query: str) -> str:
    location = extract_location_llm(query)
    if not location:
        return "I couldn't identify a location in your query. Please specify a city or place for weather information."

    tomorrow_io_api_key = os.getenv("TOMORROW_IO_API_KEY")
    base_url = "https://api.tomorrow.io/v4/weather"

    current_url = f"{base_url}/realtime?location={location}&apikey={tomorrow_io_api_key}"
    current_response = requests.get(current_url)
    
    if current_response.status_code != 200:
        return f"Error fetching current weather data: {current_response.status_code}"

    current_data = current_response.json()

    forecast_url = f"{base_url}/forecast?location={location}&apikey={tomorrow_io_api_key}"
    forecast_response = requests.get(forecast_url)
    
    if forecast_response.status_code != 200:
        return f"Error fetching forecast data: {forecast_response.status_code}"

    forecast_data = forecast_response.json()

    context = f"Current weather in {location}:\n"
    context += f"Temperature: {current_data['data']['values']['temperature']}°C\n"
    context += f"Humidity: {current_data['data']['values']['humidity']}%\n"
    context += f"Wind Speed: {current_data['data']['values']['windSpeed']} m/s\n"
    context += f"Conditions: {current_data['data']['values']['weatherCode']}\n\n"

    context += "5-day forecast:\n"
    for day in forecast_data['timelines']['daily'][:5]:
        date = datetime.fromisoformat(day['time']).strftime("%Y-%m-%d")
        context += f"{date}:\n"
        context += f"  Max Temperature: {day['values']['temperatureMax']}°C\n"
        context += f"  Min Temperature: {day['values']['temperatureMin']}°C\n"

    llm_prompt = f"""
    Given the following weather information for {location}:

    {context}

    Please provide a concise and informative summary of the current weather and 5-day forecast.
    Include any notable weather patterns or changes over the next few days.
    """

    response = chat_model.invoke(llm_prompt)
    return response.content

def use_calculator(query: str) -> str:
    expression = re.search(r'\d+[\s\d\+\-\*\/\(\)]*\d+', query)
    if expression:
        try:
            result = eval(expression.group())
            return f"The result of the calculation is: {result}"
        except:
            return "Sorry, I couldn't perform that calculation. Please check the expression and try again."
    else:
        return "I couldn't find a valid mathematical expression in your query."

def get_stock_info(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="1d")
        
        current_price = history['Close'].iloc[-1]
        market_cap_crores = info.get('marketCap', 'N/A')
        if market_cap_crores != 'N/A':
            market_cap_crores = market_cap_crores / 10000000
        
        latest_quarter = stock.quarterly_financials.columns[0]
        operating_margin = stock.quarterly_financials.loc['Operating Income', latest_quarter] / stock.quarterly_financials.loc['Total Revenue', latest_quarter] * 100

        return {
            "Ticker": ticker,
            "Current Market Price": current_price,
            "Opening Price": history['Open'].iloc[-1],
            "Closing Price": current_price,
            "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
            "Daily High": history['High'].iloc[-1],
            "Daily Low": history['Low'].iloc[-1],
            "Daily Range": f"{history['Low'].iloc[-1]} - {history['High'].iloc[-1]}",
            "Distance from 52 Week High": f"{((info.get('fiftyTwoWeekHigh', current_price) - current_price) / current_price) * 100:.2f}%",
            "Distance from 52 Week Low": f"{((current_price - info.get('fiftyTwoWeekLow', current_price)) / current_price) * 100:.2f}%",
            "Operating Profit Margin %": f"{operating_margin:.2f}%",
            "Volume": info.get('volume', 'N/A'),
            "Market Cap (in Crores)": f"₹{market_cap_crores:.2f}" if market_cap_crores != 'N/A' else 'N/A'
        }
    
    except Exception as e:
        return {"error": f"An error occurred while fetching data for {ticker}: {str(e)}"}

def use_yfinance_tool(query: str) -> str:
    match = re.search(r'\b[A-Z]+\.?[A-Z]*\b', query)
    if match:
        ticker = match.group(0)
        stock_data = get_stock_info(ticker)
        if "error" in stock_data:
            return stock_data["error"]
        else:
            context = "\n".join([f"{k}: {v}" for k, v in stock_data.items()])
            
            prompt = f"""
            Given the following stock information for {ticker}:

            {context}

            Please answer the following question:
            {query}

            Provide a concise and informative answer based on the given stock data.
            """
            
            response = chat_model.invoke(prompt)
            
            return f"Stock information for {ticker}:\n{context}\n\nAnalysis:\n{response.content}"
    else:
        return "No valid stock ticker found in the query."

def use_own_knowledge(query: str) -> str:
    response = chat_model.invoke(query)
    return response.content

tools = [
    Tool(
        name="VectorDB",
        func=query_vectordb,
        description="Useful for answering questions about physics specifically about sound, from the NCERT textbook. This tool retrieves the 5 most relevant chunks of information."
    ),
    Tool(
        name="WeatherInfo",
        func=get_weather,
        description="Useful for getting current weather information and a 5-day forecast for a specified location."
    ),
    Tool(
        name="Calculator",
        func=use_calculator,
        description="Useful for performing mathematical calculations."
    ),
    Tool(
        name="OwnKnowledge",
        func=use_own_knowledge,
        description="Useful for answering general knowledge questions or when no other tool is relevant. This tool uses the LLM's own knowledge to provide an answer."
    ),
    Tool(
        name="YFinance",
        func=use_yfinance_tool,
        description="Useful for getting detailed stock market information including historical data, dividends, market capitalization, financials, and price data for a given stock symbol and time period."
    )
]

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If no specialized tool (VectorDB, WeatherInfo, Calculator, Yfinance) is appropriate, use the OwnKnowledge tool to leverage the LLM's general knowledge.

If even after using OwnKnowledge you are not able to answer the question, inform the user that you couldn't find any relevant information.

Begin!

Question: {input}
Thought: {agent_scratchpad}""",
    tools=tools,
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)

llm_chain = LLMChain(llm=chat_model, prompt=prompt)

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=[tool.name for tool in tools],
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

class Query(BaseModel):
    text: str

@app.post("/query")
async def process_query(query: Query):
    try:
        response = agent_executor.run(query.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)