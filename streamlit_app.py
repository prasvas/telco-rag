import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=api_key
)

# Load FAISS vector DB
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_telco_index", embedding_model, allow_dangerous_deserialization=True)

# Build QA chain with source support
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=db.as_retriever())

# Streamlit UI
st.set_page_config(page_title="Telco RAG Chatbot", layout="centered")
st.title("📞 Telco RAG Chatbot")
st.markdown("Ask anything based on Telco training documents!")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Getting answer..."):
            response = qa_chain.invoke({"question": query})
        st.success("✅ Answer received!")
        st.write("### 🤖 Answer:")
        st.write(response["answer"])
        st.write("### 📄 Sources:")
        st.write(response["sources"] or "No source provided.")
    else:
        st.warning("Please enter a valid question.")
