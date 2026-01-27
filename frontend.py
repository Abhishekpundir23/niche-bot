import streamlit as st
import requests

# --- CONFIGURATION ---
# If you are running locally, use localhost
# If you deployed to Render, change this to your Render URL
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Niche Knowledge Bot", page_icon="🤖")

# --- UI HEADER ---
st.title("🤖 Niche Knowledge Bot")
st.markdown("Ask questions based on your uploaded documents.")

# --- SIDEBAR (Upload) ---
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Ingesting document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success("Document processed successfully!")
                    else:
                        st.error(f"Failed to process: {response.status_code}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
                    st.error("Make sure your Backend (main.py) is running!")

# --- CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get Bot Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"question": prompt}
                response = requests.post(f"{BACKEND_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer found.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
                st.info("Is the backend running on Port 8000?")