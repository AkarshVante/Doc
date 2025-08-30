from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, FAISS_DIR

def build_vector_store(text_chunks, progress_callback=None):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    if progress_callback:
        progress_callback("Embedding texts and building FAISS index...")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_DIR)
    if progress_callback:
        progress_callback("Saved FAISS index to disk.")
    return vector_store

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    return db
