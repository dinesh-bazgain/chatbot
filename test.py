import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
import re
import dateparser
from htmltemplates import css, user_template, bot_template

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

def get_conservation_chain(vectorStore):
    llm = ChatOllama(model="llama3")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def validate_email(email: str) -> bool:
    return re.match(r"[^@]+@[^@]+\\.[^@]+", email) is not None

def validate_phone(phone: str) -> bool:
    return re.match(r"^\\+?\\d{7,15}$", phone) is not None

def parse_date(date_str: str) -> str:
    dt = dateparser.parse(date_str)
    return dt.strftime('%Y-%m-%d') if dt else "Invalid date"

def get_agent():
    tools = [
        Tool(
            name="CollectName",
            func=lambda x: f"Thanks, {x.strip().title()}!",
            description="Collect the user's full name."
        ),
        Tool(
            name="CollectEmail",
            func=lambda x: "Valid email ✅" if validate_email(x) else "Invalid email ❌",
            description="Validate and collect user's email address."
        ),
        Tool(
            name="CollectPhone",
            func=lambda x: "Valid phone ✅" if validate_phone(x) else "Invalid phone ❌",
            description="Validate and collect user's phone number."
        ),
        Tool(
            name="CollectDate",
            func=lambda x: f"Your appointment is set for {parse_date(x)}" if parse_date(x) != "Invalid date" else "Couldn't parse the date ❌",
            description="Parse a natural language date and return in YYYY-MM-DD format."
        ),
    ]
    llm = ChatOllama(model="llama3")
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent

def handle_userinput(user_question):
    if "call me" in user_question.lower():
        agent = get_agent()
        response = agent.run(user_question)
        st.write(bot_template.replace("{{ answer }}", response), unsafe_allow_html=True)
    else:
        response = st.session_state.conservation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{ question }}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{ answer }}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chatbot")
    user_question = st.text_input("Ask questions related to the documents or request a call: ")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'PROCESS'", accept_multiple_files=True)
        if st.button("PROCESS"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorStore = get_vectorStore(text_chunks)
                st.session_state.conservation = get_conservation_chain(vectorStore)

if __name__ == '__main__':
    main()
