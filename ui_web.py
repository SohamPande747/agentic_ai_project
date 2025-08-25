import streamlit as st
import os
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import bs4



st.set_page_config(page_title="RAG with Gemini + Chroma", layout="wide")
st.title("🔎 RAG App with Gemini & Chroma")

# API key input
api_key = "AIzaSyDYVMxUbzdHaankDW_3ZVgxh7rgglEOhws"

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # Initialize models
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Vector store
    vector_store = Chroma(
        collection_name="agent-model",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    # URL input
    urls = st.text_area("Enter one or more URLs (comma-separated):",
                        "https://www.geeksforgeeks.org/linux-unix/introduction-to-linux-operating-system/")

    if st.button("Load & Index"):
        try:
            url_list = [u.strip() for u in urls.split(",") if u.strip()]
            loader = WebBaseLoader(
                web_paths=tuple(url_list),
                bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                    ["article", "div", "section", "main", "header", "footer", "aside", "p", "span"]
                ))
            )
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = text_splitter.split_documents(docs)

            # Index documents
            _ = vector_store.add_documents(documents=all_splits)
            st.success(f"Indexed {len(all_splits)} chunks from {len(url_list)} URLs!")

        except Exception as e:
            st.error(f"Error loading documents: {e}")

    # Q&A section
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            retriever = vector_store.as_retriever()
            docs = retriever.get_relevant_documents(question)

            # Prompt template
            prompt = PromptTemplate.from_template(
                "Use the following context to answer the question.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )

            context = "\n\n".join([doc.page_content for doc in docs])
            formatted_prompt = prompt.format(context=context, question=question)

            answer = llm.invoke(formatted_prompt)

            st.markdown("### Answer")
            st.write(answer.content if hasattr(answer, "content") else answer)

            with st.expander("See retrieved context"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}:**\n{d.page_content}\n")

else:
    st.info("Please enter your Google Gemini API key to get started.")
