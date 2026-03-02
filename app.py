import streamlit as st
from groq import Groq
import os, json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pypdf import PdfReader

# 1. Setup & API Keys
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
hf_client = InferenceClient(api_key=os.getenv("HF_API_KEY"))
HISTORY_FILE = "babbage_db.json"

# 2. UI Styling (Professional & Clean)
st.set_page_config(page_title="BabbageSystem AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0f1116; color: #e0e0e0; }
    .stChatMessage { border-radius: 5px; border-left: 3px solid #4A90E2; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# 3. Memory Functions
def save_memory(messages):
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f)

def load_memory():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return [{"role": "assistant", "content": "BabbageSystem AI active. Awaiting input."}]

# 4. Sidebar: Tools & Uploads
with st.sidebar:
    st.title("BabbageSystem AI")
    if st.button("Clear Memory"):
        st.session_state.messages = [{"role": "assistant", "content": "Memory wiped. System ready."}]
        save_memory(st.session_state.messages)
        st.rerun()
    
    st.divider()
    # Feature 1: Document Upload
    uploaded_pdf = st.file_uploader("Upload PDF for Assessment", type="pdf")

# 5. Initialize Session
if "messages" not in st.session_state:
    st.session_state.messages = load_memory()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Logic: Processing Input
if prompt := st.chat_input("Command the System..."):
    
    # Check for Image Request
    if prompt.lower().startswith("generate image"):
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing creative visuals..."):
                # Feature 2: Creative Image Generation (Free via HF)
                image = hf_client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
                st.image(image, caption="BabbageSystem Output")
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": f"Image generated for: {prompt}"})
                save_memory(st.session_state.messages)

    # Handle Text & Document Assessment
    else:
        pdf_context = ""
        if uploaded_pdf:
            reader = PdfReader(uploaded_pdf)
            pdf_text = "".join([page.extract_text() for page in reader.pages])
            pdf_context = f"\n\n[DOCUMENT ASSESSMENT DATA]:\n{pdf_text}"

        # Build message with context
        st.session_state.messages.append({"role": "user", "content": prompt + pdf_context})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            response_box = st.empty()
            full_text = ""
            
            # Use Llama 3.3 for high-speed reasoning
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True
            )
            
            for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    full_text += content
                    response_box.markdown(full_text + "█")
            
            response_box.markdown(full_text)
            st.session_state.messages.append({"role": "assistant", "content": full_text})
            save_memory(st.session_state.messages)