import requests
import streamlit as st


API_URL = "http://localhost:8000"

st.set_page_config(page_title="Document Q&A Assistant", layout="wide")
st.title("Document Q&A Assistant")

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Add a PDF, TXT, or Markdown document", type=["pdf", "txt", "md"])
    if uploaded_file and st.button("Index document", use_container_width=True):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
        if response.ok:
            payload = response.json()
            st.success(f"Indexed {payload['chunks_indexed']} chunks from {payload['filename']}.")
        else:
            st.error(response.json().get("detail", "Upload failed."))

question = st.chat_input("Ask a question about your uploaded documents")
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
        if response.ok:
            payload = response.json()
            st.write(payload["answer"])
            with st.expander("Sources"):
                for source in payload["sources"]:
                    page = source["page"] if source["page"] is not None else "N/A"
                    st.markdown(f"**{source['source']}** · page {page}")
                    st.caption(source["snippet"])
        else:
            st.error(response.json().get("detail", "Question answering failed."))
