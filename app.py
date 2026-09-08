"""
Streamlit Frontend for the Calc + Search Agent
------------------------------------------------
A simple chat interface that lets a user talk to the agent
(calc_search_agent.py) and see it reason through calculator
and web-search tool calls in real time.

Run with:
    streamlit run app.py
"""

import streamlit as st

# IMPORTANT: this import must match your renamed agent file's name (without ".py").
# If you named your file differently (e.g. calc_weather_agent.py), change the line below to:
#     from calc_weather_agent import build_agent
from calc_search_agent import build_agent


st.set_page_config(page_title="Calc + Search Agent", page_icon="🤖")

st.title("🤖 Calc + Search Agent")
st.caption("Ask it to do math, or ask about current things like weather, news, or prices.")


# --------------------------------------------------------------------------
# Load the agent once and cache it (so it doesn't rebuild on every message)
# --------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    return build_agent()


agent = get_agent()


# --------------------------------------------------------------------------
# Chat history (kept in session so it persists across reruns)
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------------------------------
# Chat input
# --------------------------------------------------------------------------
user_input = st.chat_input("Type your question here...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]}
                )
                answer = response["messages"][-1].content
            except Exception as exc:
                answer = f"Sorry, something went wrong: {exc}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
