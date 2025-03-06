from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
import streamlit as st
import os
from dotenv import load_dotenv

st.set_page_config(page_title = "Create FlashCards")
st.title("Flash Card Generator")
st.subheader('Generate FlashCard')

os.environ['Groq_API_KEY'] = os.getenv("Groq_API_KEY")

GROQ_API_KEY = os.getenv("Groq_API_KEY")
model = ChatGroq(model = "Deepseek-R1-Distill-Llama-70b", groq_api_key = st.secrets["Groq_API_KEY"] )

uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)


def create_vectors():
    if "vectors" in st.session_state:
        del st.session_state.vectors
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        if not uploaded_files:
            st.error("Upload File")
            return
        all_docs = []
        os.makedirs("temp", exist_ok = True)
        for file in uploaded_files:
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            loader = PyMuPDFLoader(file_path)
            all_docs.extend(loader.load())
        if not all_docs:
            st.error("No valid documents loaded. Please check your PDFs.")
            return
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        st.session_state.final_docs = st.session_state.splitter.split_documents(all_docs)
        texts = [doc.page_content for doc in st.session_state.final_docs]
        if not texts:
            st.error("No text in file Please check file")
            return
        try:
            st.session_state.vectors = FAISS.from_texts(texts, st.session_state.embeddings)
            st.success("File Uploaded Successfully")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")


mark_level = st.selectbox("Answer Length", ["Short (10-20 words)", "Long (50+ words)"])
detail_instructions = {
    "Short (10-20 words)": "Provide short answers, 10-20 words each.",
    "Long (50+ words)": "Provide long answers, 50+ words each, with detailed examples."
}
        


prompt = PromptTemplate.from_template("""
    You are an AI study assistant specializing in creating flashcards for the Anki app.  
    Generate question-answer pairs **only based on the provided context** in a simple flashcard format.  
    Do not add any new information or unnecessary details.
    Do not add any extra commentary, thoughts, or conclusions. Focus only on the Q&A.
            
    {detail_instructions}
    Try to create questions for long answers too.
    
    <Study Material>  
    {context}  
    </Study Material>  

    Format:  
    Q: [Question]
    A: [Answer]
    Ensure that each answer directly corresponds to the content in the material.
""")

if st.button("Process Documents"):
    create_vectors()

user_prompt = st.text_input("Enter your query")
if user_prompt and st.session_state.vectors:
    document_chain = create_stuff_documents_chain(model, prompt)
    retriever = create_retrieval_chain(st.session_state.vectors.as_retriever(search_kwargs={"k": 10}), document_chain)
    response = retriever.invoke({'input': user_prompt, "detail_instructions": detail_instructions[mark_level]})
    flashcard_text = response["answer"]

    lines = flashcard_text.split("\n")
    flashcards = []
    current_question = None
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            current_question = line[3:].strip()
        elif line.startswith("A:"):
            answer = line[3:].strip()
            flashcards.append((current_question, answer))
            current_question = None
    with open("flashcards.txt", "w", encoding="utf-8") as f:
        for question, answer in flashcards:
            f.write(f"{question}\t{answer}\n")
    st.write("Flashcards generated:")
    for q, a in flashcards:
        st.write(f"Q: {q}")
        st.write(f"A: {a}")
    with open("flashcards.txt", "rb") as f:
        st.download_button("Download Anki Flashcards", f, file_name="flashcards.txt")


