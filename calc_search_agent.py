"""
Calc + Search Agent
-----------------------
A lightweight conversational AI agent built with LangChain and Groq.

The agent can reason about a user's request, decide whether a tool is
required, invoke the appropriate tool, and return a final answer.

Tools available:
    1. Calculator  - evaluates basic arithmetic expressions
    2. Web Search   - retrieves current, real-world information via DuckDuckGo
                       (used for things like current weather, news, prices)

Author:  Amna Khurram
Project: LangChain Agent Development
"""

import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from tavily import TavilyClient


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
load_dotenv()  # reads keys from a local .env file (never commit that file)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0                       # 0 = deterministic, consistent tool-selection behaviour

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("calc_search_agent")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Create a .env file (see .env.example) "
        "with GROQ_API_KEY=your_key_here"
    )

if not TAVILY_API_KEY:
    raise RuntimeError(
        "TAVILY_API_KEY not found. Add TAVILY_API_KEY=your_key_here to your .env file "
        "(get a free key at app.tavily.com)"
    )


# --------------------------------------------------------------------------
# Language Model
# --------------------------------------------------------------------------
def build_llm() -> ChatGroq:
    """Initialize and return the Groq-hosted language model."""
    return ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        groq_api_key=GROQ_API_KEY,
    )


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
_ALLOWED_CHARS = set("0123456789+-*/(). ")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression.

    Use this tool whenever the user's request involves a mathematical
    calculation (e.g. addition, subtraction, multiplication, division).

    Args:
        expression: A math expression containing only numbers and the
                     operators + - * / ( ). Example: "25 * 4".

    Returns:
        The result of the calculation as a string, or an error message
        if the expression is invalid.
    """
    if not all(char in _ALLOWED_CHARS for char in expression):
        return "Invalid input: only numeric expressions are permitted."
    try:
        return str(eval(expression))
    except Exception as exc:
        logger.warning("Calculator error for input '%s': %s", expression, exc)
        return f"Could not evaluate expression: {exc}"


_tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def web_search(query: str) -> str:
    """
    Search the web for current or real-world information.

    Use this tool when the user asks about something that requires
    up-to-date knowledge the model would not otherwise have access to
    (e.g. current weather, recent news, live prices).

    Args:
        query: The search query.

    Returns:
        A summary of the top search results.
    """
    logger.info("Running web search for query: %s", query)
    try:
        results = _tavily_client.search(query=query, max_results=5)
        snippets = [r.get("content", "") for r in results.get("results", [])]
        if not snippets:
            return "No relevant search results were found."
        return "\n\n".join(snippets)
    except Exception as exc:
        logger.error("Tavily search failed for query '%s': %s", query, exc)
        return f"Search failed: {exc}"


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
def build_agent():
    """Assemble the language model and tools into a runnable agent."""
    llm = build_llm()
    return create_agent(llm, tools=[calculator, web_search])


def run_cli() -> None:
    """Start an interactive command-line session with the agent."""
    agent = build_agent()
    print("Calc + Search Agent ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Session ended.")
            break
        if not user_input:
            continue

        try:
            response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            answer = response["messages"][-1].content
            print(f"\nAgent: {answer}\n")
        except Exception as exc:
            logger.error("Agent invocation failed: %s", exc)
            print("\nAgent: Sorry, something went wrong processing that request.\n")


if __name__ == "__main__":
    run_cli()
    