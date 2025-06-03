import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import re
import dateparser
import datetime
from dateutil.relativedelta import relativedelta
from htmltemplates import css, user_template, bot_template
import pandas as pd
import os


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorStore(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore

def get_conversation_chain(vectorStore):
    llm = ChatOllama(model="llama3")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def validate_email(email: str) -> bool:
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def validate_phone(phone: str) -> bool:
    return re.match(r"^\+?\d{7,15}$", phone) is not None

def extract_date(text):
    """Parse natural language date and return in YYYY-MM-DD format"""
    settings = {
        'PREFER_DATES_FROM': 'future',
        'RELATIVE_BASE': datetime.datetime.now()
    }
    parsed_date = dateparser.parse(text, settings=settings)
    
    if parsed_date:
        return parsed_date.strftime('%Y-%m-%d')
    return None

def start_contact_flow():
    st.session_state.contact_step = "name"
    st.session_state.contact_info = {}
    st.session_state.chat_history.append({"role": "bot", "content": "Sure! What's your full name?"})

def save_contact_info(info):
    file_path = "contact_data.csv"
    df = pd.DataFrame([info])
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)


def handle_userinput(user_question):
    user_question = user_question.strip()
    if not user_question:
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # If contact flow is active
    if st.session_state.contact_step:
        step = st.session_state.contact_step
        info = st.session_state.contact_info

        if step == "name":
            info['name'] = user_question.title()
            st.session_state.contact_step = "email"
            bot_msg = f"Thanks, {info['name']}! Now, what's your email?"
            st.session_state.chat_history.append({"role": "bot", "content": bot_msg})

        elif step == "email":
            if validate_email(user_question):
                info['email'] = user_question
                st.session_state.contact_step = "phone"
                bot_msg = "Great! Please provide your phone number (with country code)."
                st.session_state.chat_history.append({"role": "bot", "content": bot_msg})
            else:
                bot_msg = "Hmm, that email looks invalid. Try again!"
                st.session_state.chat_history.append({"role": "bot", "content": bot_msg})

        elif step == "phone":
            if validate_phone(user_question):
                info['phone'] = user_question
                st.session_state.contact_step = "date"
                bot_msg = "Perfect! When should we contact you?"
                st.session_state.chat_history.append({"role": "bot", "content": bot_msg})
            else:
                bot_msg = "That phone number doesn't look right. Please try again."
                st.session_state.chat_history.append({"role": "bot", "content": bot_msg})
        
        # NEW STEP: Date collection
        elif step == "date":
            preferred_date = extract_date(user_question)
            if preferred_date:
                st.session_state.contact_step = None
                bot_msg = (f"Awesome! We'll contact you on {preferred_date}, {info['name']} "
                          f"at {info['email']} or {info['phone']}.")
                st.session_state.chat_history.append({"role": "bot", "content": bot_msg})
                st.session_state.contact_info['date'] = preferred_date
                if preferred_date:
                    info['date'] = preferred_date
                    save_contact_info(info)

            else:
                bot_msg = "Sorry, I didn't understand that date. Please try again (e.g., 'tomorrow', 'next Tuesday')"
                st.session_state.chat_history.append({"role": "bot", "content": bot_msg})

    else:
        # Normal Q&A with PDFs
        if "call me" in user_question.lower() or "contact me" in user_question.lower():
            start_contact_flow()
        elif "appointment" in user_question.lower() or "book" in user_question.lower():
            start_contact_flow()
        else:
            # Make sure conversation chain is ready
            if st.session_state.conversation is None:
                st.session_state.chat_history.append({"role": "bot", "content": "Please upload and process documents first."})
            else:
                response = st.session_state.conversation({'question': user_question})
                bot_reply = response['answer']
                st.session_state.chat_history.append({"role": "bot", "content": bot_reply})
def main():
    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "contact_step" not in st.session_state:
        st.session_state.contact_step = None

    if "contact_info" not in st.session_state:
        st.session_state.contact_info = {}

    st.header("Chatbot")

    with st.form(key='chat_form', clear_on_submit=True):
        user_question = st.text_input(
            "Ask questions or say 'call me' to request contact:",
            key="user_input",
            placeholder="Type your message here...",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Send")
    
    if submitted and user_question.strip():
        handle_userinput(user_question.strip())

    # Display chat history container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(user_template.replace("{{ question }}", message["content"]), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{ answer }}", message["content"]), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'PROCESS'", accept_multiple_files=True)
        if st.button("PROCESS"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorStore = get_vectorStore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorStore)
                    st.success("Documents processed! You can now ask questions.")
        
        if st.session_state.contact_info:
            st.subheader("Collected Contact Info")
            for key, value in st.session_state.contact_info.items():
                st.write(f"{key.title()}: {value}")

if __name__ == '__main__':
    main()