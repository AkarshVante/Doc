from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_pdf_text(pdf_files):
    """Extract text from a list of file-like objects (Streamlit uploaded files)."""
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                p = page.extract_text()
                if p:
                    text += p + "\n"
        except Exception:
            # skip unreadable files
            continue
    return text

def get_text_chunks(text, chunk_size=3000, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
