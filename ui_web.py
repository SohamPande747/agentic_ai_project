import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
from model import ask_question, add_websites

st.set_page_config(page_title="RAG Demo", layout="centered")
st.title("RAG Demo (Streamlit UI)")

# --- Add Website ---
st.subheader("Add Websites to Knowledge Base")
with st.form("add_websites"):
    urls = st.text_area("Enter URLs (one per line):")
    submitted = st.form_submit_button("Add")
    if submitted and urls.strip():
        url_list = [u.strip() for u in urls.splitlines() if u.strip()]
        try:
            n = add_websites(url_list)
            st.success(f"Added {n} new chunks.")
        except Exception as e:
            st.error(str(e))

# --- Ask Question ---
st.subheader("Ask a Question")
question = st.text_input("Your question:")
if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Thinking..."):
            try:
                result = ask_question(question)
                answer = result.get("answer", "No answer generated.")
                st.subheader("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(str(e))
    else:
        st.warning("Please enter a question first.")
