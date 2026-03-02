import streamlit as st
import os, fitz, requests, io
from groq import Groq
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="BabbageSystem Pro", layout="wide")
st.title("BabbageSystem Pro: AI + Docs + Art")

# 2. Initialize API Clients
# Make sure these match your Streamlit Secrets exactly!
client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))
HF_TOKEN = st.secrets.get("HF_TOKEN")

# 3. Persistent History Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar UI
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    
    # Document Upload
    st.header("📄 Document Assistant")
    uploaded_file = st.file_uploader("Upload a PDF for analysis", type="pdf")
    doc_text = ""
    if uploaded_file:
        try:
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                for page in doc:
                    doc_text += page.get_text()
            st.success("PDF Content Loaded!")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

# 4. Image Generation Function (Updated for better reliability)
def generate_image(prompt):
    # Using FLUX.1-schnell (very fast)
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    
    if response.status_code != 200:
        raise Exception(f"HF API Error {response.status_code}: {response.text}")
        
    return response.content

# 5. Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"])
        else:
            st.markdown(message["content"])

# 6. Chat Input Logic
if prompt := st.chat_input("Ask a question or say 'draw [something]...'"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # IMAGE GENERATION MODE
    if any(word in prompt.lower() for word in ["draw", "image", "generate", "picture"]):
        with st.chat_message("assistant"):
            with st.spinner("🎨 Painting your request..."):
                try:
                    img_bytes = generate_image(prompt)
                    # Use PIL to ensure it's a valid image
                    image = Image.open(io.BytesIO(img_bytes))
                    st.image(image)
                    st.session_state.messages.append({"role": "assistant", "image": img_bytes})
                except Exception as e:
                    st.error(f"Image generation failed: {e}")
                    st.info("Tip: Ensure your HF_TOKEN has 'Write' access or 'Inference' permissions.")
    
    # DOCUMENT / TEXT MODE
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                # Pass first 4000 characters of PDF as context
                context = f"\n\nContext from PDF: {doc_text[:4000]}" if doc_text else ""
                try:
                    # Prepare history for Groq (excluding image data)
                    history = [
                        {"role": m["role"], "content": m["content"]} 
                        for m in st.session_state.messages if "content" in m
                    ]
                    
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": f"You are BabbageSystem. {context}"},
                            *history
                        ],
                    )
                    ans = response.choices[0].message.content
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"Text error: {e}")