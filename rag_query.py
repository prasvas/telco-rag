from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# ✅ Load your FAISS vector DB
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_telco_index", embedding_model, allow_dangerous_deserialization=True)

# ✅ Use Gemini via API Key (NOT Vertex AI)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key="AIzaSyCE7Q-afzTRxxgfEfKmrbNHqO_9zjw80Zo"
)

# ✅ Build QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# ✅ Ask a question
query = "How do I upgrade my mobile plan?"
answer = qa_chain.invoke({"query": query})
print("🤖 Answer:", answer)
