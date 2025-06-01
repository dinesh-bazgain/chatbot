import streamlit as st

def main():
    st.set_page_config(
        page_title="Chatbot",
        page_icon="🤖"
    )

    st.header("Chatbot")

    with st.sidebar:
        user_input = st.text_input("Your question: ", key="user_input")

if  __name__ == '__main__':
    main()