import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🛠 Replace below with your actual full path
pdf_path = r"C:\Users\prasvas\Desktop\Personal Prasanna\GCP\telco-rag\test.pdf"

# Check working directory
print("Working directory:", os.getcwd())

# Check if file exists
print("Checking this file path:", pdf_path)
print("File exists:", os.path.exists(pdf_path))

# Proceed only if file exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ File not found at: {pdf_path}")

# Load & chunk PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print(f"✅ Total Chunks: {len(chunks)}")
print(chunks[0])

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Use the MiniLM model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build the FAISS index
db = FAISS.from_documents(chunks, embedding_model)
db.save_local("faiss_telco_index")
print("✅ FAISS index saved.")