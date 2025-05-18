import os
import tempfile
import pytesseract
import pdfplumber
from PIL import Image
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gtts import gTTS
import streamlit as st
import re
import nltk
import random
import torch  # Import torch

# Download NLTK resources if not already downloaded
# nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

# Set a writable download directory
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)

nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

def setup_nltk():
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)

setup_nltk()


# Set NLTK to use this directory
# nltk.data.path.append(nltk_data_dir)

# # Download required resources to this directory
# nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
# nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)

# os.makedirs(nltk_data_dir, exist_ok=True)
# nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
# nltk.data.path.append(nltk_data_dir)

# nltk_data_dir = "/tmp/nltk_data"
# os.makedirs(nltk_data_dir, exist_ok=True)
# nltk.data.path.append(nltk_data_dir)

# # nltk.download('punkt', download_dir=nltk_data_dir)
# # nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
# nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
# nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)



# Load models

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=torch.device("cpu"))
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_dbs = {}  # Dictionary to store multiple vector databases, keyed by document title
extracted_texts = {} # Dictionary to store extracted text, keyed by document title
current_doc_title = None

# ---------------------------------------------
# Extract text from PDF or Image
# ---------------------------------------------
def extract_text(uploaded_file):
    global current_doc_title
    current_doc_title = uploaded_file.name
    suffix = uploaded_file.name.lower()
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    text = ""
    if suffix.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    else:
        try:
            text = pytesseract.image_to_string(Image.open(path))
        except Exception as e:
            st.error(f"Error during OCR for {uploaded_file.name}: {e}")
            text = ""

    os.remove(path)
    return text.strip()

# ---------------------------------------------
# Store Embeddings in FAISS
# ---------------------------------------------
def store_vector(text):
    global vector_dbs, current_doc_title
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([text])
    for doc in docs:
        doc.metadata = {"title": current_doc_title}  # Add document title as metadata

    if current_doc_title in vector_dbs:
        vector_dbs[current_doc_title].add_documents(docs)  # Append to existing DB
    else:
        vector_dbs[current_doc_title] = FAISS.from_documents(docs, embedding_model)

# ---------------------------------------------
# Summarize Text
# ---------------------------------------------
def summarize(text):
    if len(text.split()) < 100:
        return "Text too short to summarize."
    chunks = [text[i : i + 1024] for i in range(0, len(text), 1024)]
    summaries = []
    for chunk in chunks:
        max_len = int(len(chunk.split()) * 0.6)
        max_len = max(30, min(max_len, 150))
        try:
            summary = summarizer(chunk, max_length=max_len, min_length=20)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            return "An error occurred during summarization."
    return " ".join(summaries)

# ---------------------------------------------
# Question Answering
# ---------------------------------------------
def topic_search(question, doc_title=None):
    global vector_dbs
    if not vector_dbs:
        st.warning("Please upload and process a file first in the 'Upload & Extract' tab.")
        return ""

    try:
        if doc_title and doc_title in vector_dbs:
            retriever = vector_dbs[doc_title].as_retriever(search_kwargs={"k": 3})
        else:
            combined_docs = []
            for db in vector_dbs.values():
                combined_docs.extend(db.get_relevant_documents(question))
            if not combined_docs:
                 return "No relevant information found across uploaded documents."
            temp_db = FAISS.from_documents(combined_docs, embedding_model)
            retriever = temp_db.as_retriever(search_kwargs={"k": 3})

        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        answer = qa_pipeline(question=question, context=context)["answer"]
        return answer.strip()
    except Exception as e:
        st.error(f"Error during question answering: {e}")
        return "An error occurred while trying to answer the question."

# ---------------------------------------------
# Flashcard Generation
# ---------------------------------------------
def generate_flashcards(text):
    flashcards = []
    seen_terms = set()
    sentences = nltk.sent_tokenize(text)

    for i, sent in enumerate(sentences):
        words = nltk.word_tokenize(sent)
        tagged_words = nltk.pos_tag(words)

        potential_terms = [word for word, tag in tagged_words if tag.startswith('NN') or tag.startswith('NP')]

        for term in potential_terms:
            if term in seen_terms:
                continue

            defining_patterns = [r"\b" + re.escape(term) + r"\b\s+is\s+(?:a|an|the)\s+(.+?)(?:\.|,|\n|$)",
                                 r"\b" + re.escape(term) + r"\b\s+refers\s+to\s+(.+?)(?:\.|,|\n|$)",
                                 r"\b" + re.escape(term) + r"\b\s+means\s+(.+?)(?:\.|,|\n|$)",
                                 r"\b" + re.escape(term) + r"\b,\s+defined\s+as\s+(.+?)(?:\.|,|\n|$)",
                                 r"\b" + re.escape(term) + r"\b:\s+(.+?)(?:\.|,|\n|$)"]

            potential_definitions = []
            for pattern in defining_patterns:
                match = re.search(pattern, sent, re.IGNORECASE)
                if match and len(match.groups()) >= 1:
                    potential_definitions.append(match.group(1).strip())

            for definition in potential_definitions:
                if 2 <= len(definition.split()) <= 30:
                    flashcards.append({"term": term, "definition": definition})
                    seen_terms.add(term)
                    break

            if term not in seen_terms and i > 0:
                prev_sent = sentences[i-1]
                defining_patterns_prev = [r"The\s+\b" + re.escape(term) + r"\b\s+is\s+(.+?)(?:\.|,|\n|$)",
                                          r"This\s+\b" + re.escape(term) + r"\b\s+refers\s+to\s+(.+?)(?:\.|,|\n|$)",
                                          r"It\s+means\s+the\s+\b" + re.escape(term) + r"\b\s+(.+?)(?:\.|,|\n|$)"]
                for pattern in defining_patterns_prev:
                    match = re.search(pattern, prev_sent, re.IGNORECASE)
                    if match and term in sent and len(match.groups()) >= 1:
                        definition = match.group(1).strip()
                        if 2 <= len(definition.split()) <= 30:
                            flashcards.append({"term": term, "definition": definition})
                            seen_terms.add(term)
                            break

    return flashcards

# ---------------------------------------------
# Text to Speech
# ---------------------------------------------
def read_aloud(text):
    try:
        tts = gTTS(text)
        audio_path = os.path.join(tempfile.gettempdir(), "summary.mp3")
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Error during text-to-speech: {e}")
        return None

# ---------------------------------------------
# Quiz Generation and Handling
# ---------------------------------------------
def generate_quiz_questions(text, num_questions=5):
    flashcards = generate_flashcards(text) # Reuse flashcard logic for potential terms/definitions
    if not flashcards:
        return []

    questions = []
    used_indices = set()
    num_available = len(flashcards)

    while len(questions) < num_questions and len(used_indices) < num_available:
        index = random.randint(0, num_available - 1)
        if index in used_indices:
            continue
        used_indices.add(index)
        card = flashcards[index]
        correct_answer = card['term']
        definition = card['definition']

        # Generate incorrect answers (very basic for now)
        incorrect_options = random.sample([c['term'] for i, c in enumerate(flashcards) if i != index], 3)
        options = [correct_answer] + incorrect_options
        random.shuffle(options)

        questions.append({
            "question": f"What is the term for: {definition}",
            "options": options,
            "correct_answer": correct_answer,
            "user_answer": None # To store user's choice
        })

    return questions

def display_quiz(questions):
    st.session_state.quiz_questions = questions
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    for i, q in enumerate(st.session_state.quiz_questions):
        st.subheader(f"Question {i + 1}:")
        st.write(q["question"])
        st.session_state.user_answers[i] = st.radio(f"Answer for Question {i + 1}", q["options"])
    st.button("Submit Quiz", on_click=submit_quiz)

def submit_quiz():
    st.session_state.quiz_submitted = True

def grade_quiz():
    if st.session_state.quiz_submitted:
        score = 0
        for i, q in enumerate(st.session_state.quiz_questions):
            user_answer = st.session_state.user_answers.get(i)
            if user_answer == q["correct_answer"]:
                score += 1
                st.success(f"Question {i + 1}: Correct!")
            else:
                st.error(f"Question {i + 1}: Incorrect. Correct answer was: {q['correct_answer']}")
        st.write(f"## Your Score: {score} / {len(st.session_state.quiz_questions)}")

# ---------------------------------------------
# Streamlit Interface with Tabs
# ---------------------------------------------
st.title("📘 AI Study Assistant")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Extract", "Summarize", "Question Answering", "Interactive Learning", "Quiz"])

with tab1:
    st.header("📤 Upload and Extract Text")
    uploaded_files = st.file_uploader("Upload multiple PDF or Image files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"Extracting text from {file.name}..."):
                extracted_text = extract_text(file)
            if extracted_text:
                extracted_texts[file.name] = extracted_text
                st.success(f"Text Extracted Successfully from {file.name}!")
                with st.expander(f"View Extracted Text from {file.name}"):
                    st.text_area("Extracted Text", value=extracted_text[:3000], height=400)
                store_vector(extracted_text)
            else:
                st.warning(f"Could not extract any text from {file.name}.")

with tab2:
    st.header("📝 Summarize Text")
    doc_titles = list(vector_dbs.keys())
    if doc_titles:
        selected_doc_title_summary = st.selectbox("Summarize document:", doc_titles)
        if st.button("Generate Summary"):
            if selected_doc_title_summary in extracted_texts:
                with st.spinner(f"Summarizing {selected_doc_title_summary}..."):
                    summary = summarize(extracted_texts[selected_doc_title_summary])
                st.subheader("Summary")
                st.write(summary)
                audio_path = read_aloud(summary)
                if audio_path:
                    st.audio(audio_path)
            else:
                st.warning(f"Original text for {selected_doc_title_summary} not found. Please re-upload.")
    else:
        st.info("Please upload and extract a file in the 'Upload & Extract' tab first.")

with tab3:
    st.header("❓ Question Answering")
    doc_titles = list(vector_dbs.keys())
    if doc_titles:
        doc_title = st.selectbox("Search within document:", ["All Documents"] + doc_titles)
        question = st.text_input("Ask a question about the content:")
        if question:
            with st.spinner("Searching for answer..."):
                if doc_title == "All Documents":
                    answer = topic_search(question)
                else:
                    answer = topic_search(question, doc_title=doc_title)
            if answer:
                st.subheader("Answer:")
                st.write(answer)
            else:
                st.warning("Could not find an answer in the selected document(s).")
    else:
        st.info("Please upload and extract a file in the 'Upload & Extract' tab first.")

with tab4:
    st.header("🧠 Interactive Learning: Flashcards")
    doc_titles = list(extracted_texts.keys())
    if doc_titles:
        selected_doc_title_flashcard = st.selectbox("Generate flashcards from document:", doc_titles)
        if st.button("Generate Flashcards"):
            if selected_doc_title_flashcard in extracted_texts:
                with st.spinner(f"Generating flashcards from {selected_doc_title_flashcard}..."):
                    flashcards = generate_flashcards(extracted_texts[selected_doc_title_flashcard])
                if flashcards:
                    st.subheader("Flashcards")
                    for i, card in enumerate(flashcards):
                        with st.expander(f"Card {i+1}"):
                            st.markdown(f"*Term:* {card['term']}")
                            st.markdown(f"*Definition:* {card['definition']}")
                else:
                    st.info("No flashcards could be generated from this document using the current method.")
            else:
                st.warning(f"Original text for {selected_doc_title_flashcard} not found. Please re-upload.")
    else:
        st.info("Please upload and extract a file in the 'Upload & Extract' tab first.")

with tab5:
    st.header("📝 Quiz Yourself!")
    doc_titles = list(extracted_texts.keys())
    if doc_titles:
        selected_doc_title_quiz = st.selectbox("Generate quiz from document:", doc_titles)
        if selected_doc_title_quiz in extracted_texts:
            text_for_quiz =extracted_texts[selected_doc_title_quiz]
            if "quiz_questions" not in st.session_state:
                st.session_state.quiz_questions = generate_quiz_questions(text_for_quiz)

            if st.session_state.quiz_questions:
                display_quiz(st.session_state.quiz_questions)
                if st.session_state.quiz_submitted:
                    grade_quiz()

                if st.button("Refresh Questions"):
                    st.session_state.quiz_questions = generate_quiz_questions(text_for_quiz)
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}
                    st.rerun() # Force a re-render to show new questions
            else:
                st.info("Could not generate quiz questions from the current document.")
        else:
            st.warning(f"Original text for {selected_doc_title_quiz} not found. Please re-upload.")
    else:
        st.info("Please upload and extract a file in the 'Upload & Extract' tab first.")
