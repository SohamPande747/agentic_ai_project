# model.py
# RAG pipeline using LangChain + Google Gemini (chat + embeddings) + Chroma

import os
from typing import List, TypedDict

from dotenv import load_dotenv
import asyncio

# Ensure an event loop exists (needed for grpc.aio in Streamlit threads)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# --- Load env and validate API key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment. "
        "Set it in a .env file or as a system environment variable."
    )

# --- LLM (chat) ---
from langchain.chat_models import init_chat_model

# Choose a fast chat model; adjust if you need higher quality
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# --- Embeddings ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# --- Vector store (Chroma) ---
from langchain_chroma import Chroma

PERSIST_DIR = "./chroma_langchain_db"
vector_store = Chroma(
    collection_name="agent-model",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# --- Loaders / splitting ---
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Prompting ---
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# --- LangGraph state graph ---
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    # You can tune k as needed
    retrieved_docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    formatted_prompt = prompt.format(
        question=state["question"],
        context=docs_content if docs_content else "(no relevant context found)"
    )
    response = llm.invoke(formatted_prompt)
    # Some LLMs return `AIMessage` with `.content`; others a string. Handle both.
    answer_text = getattr(response, "content", response)
    return {"answer": answer_text}


# Build the graph explicitly (clearer than add_sequence for named nodes)
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()


# ---------------- Public Functions ----------------
def ask_question(question: str):
    """Run the retrieval+generation graph for a question."""
    return graph.invoke({"question": question})


def add_websites(urls: List[str]):
    """
    Load and add websites to the vector store.
    Returns the number of chunks indexed.
    """
    if not urls:
        return 0

    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer([
                "article", "div", "section", "main", "header",
                "footer", "aside", "p", "span"
            ])
        )
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = splitter.split_documents(docs)

    if not splits:
        return 0

    vector_store.add_documents(splits)
    # Persist so the new chunks survive interpreter restarts
    vector_store.persist()
    return len(splits)
