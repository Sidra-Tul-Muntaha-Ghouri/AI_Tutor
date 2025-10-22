import os
import tempfile
import pytesseract
import pdfplumber
from PIL import Image
import streamlit as st
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from gtts import gTTS
import base64

# Load lightweight models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = None

# -------------------------------
# ✅ Extract text from PDF/Image
# -------------------------------
def extract_text(file):
    suffix = file.name.lower()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        path = tmp.name

    if suffix.endswith(".pdf"):
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    else:
        text = pytesseract.image_to_string(Image.open(path))

    os.remove(path)
    return text.strip()

# -------------------------------
# ✅ Embed and Store Vector
# -------------------------------
def store_vector(text):
    global vector_db
    doc = Document(page_content=text)
    vector_db = FAISS.from_documents([doc], embedding_model)

# -------------------------------
# ✅ Summarize Text
# -------------------------------
def summarize(text):
    if len(text.split()) < 100:
        return "Text too short to summarize."
    result = summarizer(text[:1024])[0]["summary_text"]
    return result

# -------------------------------
# ✅ Generate Questions
# -------------------------------
def generate_questions(text):
    desc_q = f"What is the main idea of this passage?\n{text[:200]}"
    mcq = f"Which of the following best describes the topic?\nA. ...\nB. ...\nC. ...\nAnswer: ..."
    return desc_q, mcq

# -------------------------------
# ✅ Ask a Question
# -------------------------------
def topic_search(question):
    context = vector_db.as_retriever().get_relevant_documents(question)[0].page_content
    return qa_pipeline(question=question, context=context)['answer']

# -------------------------------
# ✅ Text-to-Speech with download
# -------------------------------
def read_aloud(text):
    tts = gTTS(text)
    path = "audio.mp3"
    tts.save(path)
    with open(path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    os.remove(path)

# -------------------------------
# ✅ Streamlit UI
# -------------------------------
st.title("🧠 AI Document Analyzer")

uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    st.success("File uploaded successfully!")
    text = extract_text(uploaded_file)
    
    st.subheader("📄 Extracted Text:")
    st.text(text[:500] + ("..." if len(text) > 500 else ""))

    summary = summarize(text)
    st.subheader("📝 Summary:")
    st.write(summary)

    store_vector(text)

    desc_q, mcq = generate_questions(text)
    st.subheader("❓ Descriptive Question:")
    st.write(desc_q)

    st.subheader("📘 MCQ Sample:")
    st.write(mcq)

    user_question = st.text_input("Ask a question from the document:")
    if user_question:
        answer = topic_search(user_question)
        st.subheader("💡 Answer:")
        st.write(answer)

    if st.button("🔊 Read Summary Aloud"):
        read_aloud(summary)
