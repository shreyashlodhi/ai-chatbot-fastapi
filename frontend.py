# Step-1 : Setup UI with streamlit (model provider, model, web_search, system prompt, query)
import streamlit as st

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with the AI Agents!")

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["GPT-4o-mini"]

provider=st.radio("Select Provider:", ("Groq","OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_serach = st.checkbox("Allow Web Search")
user_query = st.text_area("Enter your query: ", height=150, placeholder="Ask Anything!")

API_URL="http://127.0.0.1:9998/chat"
# API_URL = "http://0.0.0.0:8080/chat"

if st.button("Ask Agent!"):
    if user_query.strip():

        # Step-2 : Connect with backend via URL
        import requests

        payload={
            "model_name": selected_model,
            "model_provider": provider,
            # system_prompt: str
            "messages": [user_query],
            "allow_search": allow_web_serach
        }

        response=requests.post(API_URL, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response_data}")

