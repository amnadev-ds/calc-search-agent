# Calc Search Agent

A conversational AI agent built with LangChain that combines calculator and web search capabilities, powered by the Groq API.

## Features
- Natural language calculator tool for arithmetic queries
- Web search integration (Tavily API) for real-time information
- CLI-based interaction interface
- Streamlit frontend for a simple chat UI

## Tech Stack
- Python
- LangChain
- Groq API (LLM inference)
- Tavily API (web search)
- Streamlit (frontend)

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your `GROQ_API_KEY` and `TAVILY_API_KEY`
4. Run the CLI agent: `python calc_search_agent.py`
5. Or run the Streamlit app: `streamlit run app.py`

## About
Built as part of a Data Science internship/coursework project, combining tool-use agents with real-time search grounding.
