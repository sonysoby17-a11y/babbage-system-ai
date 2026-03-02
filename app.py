import streamlit as st
import os, fitz
from groq import Groq

# 1. Page Configuration
st.set_page_config(page_title="BabbageSystem AI", layout="centered")
st.title("🤖 BabbageSystem AI")
st.caption("Document Assistant & Chat")

# 2. Initialize Groq Client
# Ensure 'GROQ_API_KEY' is in your Streamlit Secrets
client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

# 3. Persistent History Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF and History Controls
with st.sidebar:
    st.header("⚙️ Options")
    
    # Clear Chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Download Chat History
    if st.session_state.messages:
        chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.download_button("📥 Download Transcript", chat_text, file_name="chat_history.txt")

    st.divider()
    
    # Document Upload
    st.header("📄 PDF Upload")
    uploaded_file = st.file_uploader("Upload a document for analysis", type="pdf")
    doc_text = ""
    if uploaded_file:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                doc_text += page.get_text()
        st.success("Document loaded successfully!")

# 4. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Main Chat Logic
if prompt := st.chat_input("Ask a question about your doc or anything else..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare context from PDF (up to 6000 chars for better accuracy)
            pdf_context = f"\n\n[DOCUMENT CONTENT]:\n{doc_text[:6000]}" if doc_text else ""
            
            try:
                # Build message list for Groq
                messages_for_api = [
                    {"role": "system", "content": f"You are BabbageSystem, a helpful AI. Use the following document context if relevant to the user's question. {pdf_context}"},
                    *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                ]
                
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages_for_api,
                )
                
                ans = response.choices[0].message.content
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                
            except Exception as e:
                st.error(f"Error: {e}")