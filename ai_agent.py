# Step-1 : Setup API Keys for Groq and Tavily
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
# OPEN_API_KEY=os.environ.get("OPEN_API_KEY")

# Step-2 : Setup LLM & Tools
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# openai_llm=ChatOpenAI(model=)
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearch(max_results=2)

# Step-3 : Setup AI Agent with Search tool Functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id,query,allow_search, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    # elif provider=="OpenAI":
    #     llm=ChatOpenAI(model=llm_id)

    tools=[TavilySearch(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=groq_llm,
        tools=[search_tool],
    )


    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message,AIMessage)]
    return ai_messages[-1]