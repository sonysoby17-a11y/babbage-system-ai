import streamlit as st
from groq import Groq
import os

# Page Config
st.set_page_config(page_title="BabbageSystem AI", page_icon="🤖")
st.title("🤖 BabbageSystem AI")

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can BabbageSystem help you today?"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are BabbageSystem, a helpful AI."},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            ],
        )
        full_response = response.choices[0].message.content
        
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    except Exception as e:
        st.error(f"Error: {e}")