import streamlit as st
import os, fitz, io
from groq import Groq
from PIL import Image
from huggingface_hub import InferenceClient

# 1. Page Configuration
st.set_page_config(page_title="BabbageSystem Pro", layout="wide")
st.title("BabbageSystem Pro: AI + Docs + Art")

# 2. Initialize API Clients
client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))
# The InferenceClient automatically uses the 2026 router
hf_client = InferenceClient(api_key=st.secrets.get("HF_TOKEN"))

# 3. Persistent History Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar UI
with st.sidebar:
    st.header("⚙️ Controls")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Download Chat History
    if st.session_state.messages:
        chat_text = "\n".join([f"{m['role']}: {m.get('content', '[Image]')}" for m in st.session_state.messages])
        st.download_button("📥 Download Transcript", chat_text, file_name="chat_history.txt")

    st.divider()
    
    st.header("📄 Document Assistant")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    doc_text = ""
    if uploaded_file:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                doc_text += page.get_text()
        st.success("PDF Content Loaded!")

# 4. Modern Image Generation (Using the 2026 Router)
def generate_image(prompt):
    # This calls the black-forest-labs FLUX model via the new HF Router
    image = hf_client.text_to_image(
        prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )
    # Convert PIL Image to bytes for storage
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# 5. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"])
        else:
            st.markdown(message["content"])

# 6. Chat Input Logic
if prompt := st.chat_input("Ask a question or say 'draw [something]...'"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # IMAGE MODE
    if any(word in prompt.lower() for word in ["draw", "image", "generate", "art"]):
        with st.chat_message("assistant"):
            with st.spinner("🎨 Painting with FLUX..."):
                try:
                    img_bytes = generate_image(prompt)
                    st.image(img_bytes)
                    st.session_state.messages.append({"role": "assistant", "image": img_bytes})
                except Exception as e:
                    st.error(f"Image Error: {e}")
    
    # DOCUMENT / TEXT MODE
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                context = f"\n\nContext from PDF: {doc_text[:4000]}" if doc_text else ""
                try:
                    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if "content" in m]
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": f"You are BabbageSystem. {context}"}, *history],
                    )
                    ans = response.choices[0].message.content
                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"Text error: {e}")