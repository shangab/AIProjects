import streamlit as st
from helpers import LLMTool

with st.sidebar:
    st.title('Database connection Params')
    host = st.text_input('Host', 'localhost')
    port = st.text_input('Port', '3306')
    username = st.text_input('Username', 'root')
    password = st.text_input('Password', 'root', type='password')
    database = st.text_input('Database', 'MSD')
    connect_btn = st.button('Connect to DB')

    if connect_btn:
        db_uri = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
        with st.spinner('Connecting to DB...'):
            st.session_state.llm = LLMTool(db_uri=db_uri)
            st.success('Connected!')


if not 'llm' in st.session_state:
    st.warning('Please connect to the database first')
else:
    st.image('assets/logo.png', width=150)
    st.title('Talk to Databases')
    st.subheader(
        'Ask a question in any language to this DB')

    question = st.chat_input('Ask a Question')
    if question:
        query, results, answer = st.session_state.llm.runall(question)
        st.subheader('Answer:')
        st.write(answer)
        st.subheader('SQL Query:')
        st.code(query)
