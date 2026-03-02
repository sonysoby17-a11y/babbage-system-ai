import streamlit as st
import os, fitz, requests
from groq import Groq

# Page Config
st.set_page_config(page_title="BabbageSystem Pro", layout="wide")
st.title("BabbageSystem Pro: AI + Docs + Art")

# Initialize API Clients
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
HF_TOKEN = os.environ.get("HF_TOKEN") # For Image Generation

# Sidebar for Document Upload
with st.sidebar:
    st.header("📄 Document Assistant")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    doc_text = ""
    if uploaded_file:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                doc_text += page.get_text()
        st.success("Document analyzed!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to generate images via Hugging Face
def generate_image(prompt):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.content

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"])
        else:
            st.markdown(message["content"])

# Main Input
if prompt := st.chat_input("Ask about the doc or say 'draw...'"):
    st.chat_message("user").markdown(prompt)
    
    # IMAGE MODE: If user starts with "draw" or "image"
    if prompt.lower().startswith(("draw", "image")):
        with st.chat_message("assistant"):
            with st.spinner("🎨 Creating your masterpiece..."):
                img_bytes = generate_image(prompt)
                st.image(img_bytes)
                st.session_state.messages.append({"role": "assistant", "image": img_bytes})
    
    # TEXT MODE: Answer based on PDF or general knowledge
    else:
        context = f"\n\nContext from document: {doc_text[:5000]}" if doc_text else ""
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"You are BabbageSystem. Answer using this document context if available: {context}"},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if "content" in m]
                ],
            )
            ans = response.choices[0].message.content
            with st.chat_message("assistant"):
                st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.error(f"Error: {e}")