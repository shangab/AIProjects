import streamlit as st
from helpers import VectorDatabase, LLMHelper
from PyPDF2 import PdfReader


with st.sidebar:
    db = None
    llm = None
    st.title('Upload Knoelwdge Files')
    user_hf_token = st.text_input(
        "Hugging Face API Key", type="password")
    user_openai_key = st.text_input("OpenAI API Key", type="password")

    if user_openai_key is not None and user_hf_token is not None:
        st.secrets.items() ["HUGGINGFACE_ACCESS_TOKEN"] = user_hf_token
        st.secrets["OPENAI_API_KEY"] = user_openai_key
        db = VectorDatabase()
        llm = LLMHelper()
        chunks = db.get_chunks_count() if db is not None else 0
        st.subheader(f'Database contains {chunks} Vectors')

        pdf_files = st.file_uploader(
            "Choose your files", accept_multiple_files=True)
        update_btn = st.button("Update Database")
        if update_btn:
            if db is None:
                st.warning("Please provide Hugging Face API Key")
            else:
                if not pdf_files:
                    st.warning("Please upload files first")
                else:
                    with st.spinner('Reading Files...'):
                        for pdf in pdf_files:
                            with st.spinner(f'Processing pdf: {pdf.name}...'):
                                reader = PdfReader(pdf)
                                texts = []
                                metas = []
                                page_count = 1
                                for page in reader.pages:
                                    texts.append(
                                        page.extract_text().replace("\n", " "))
                                    metas.append(
                                        {"pdf_name": pdf.name,  "page_number": page_count})
                                    page_count += 1
                                with st.spinner(f'Updating database with {page_count} pages...'):
                                    db.update_db(texts=texts, metadatas=metas)

                        st.success("Database Updated")
                        updated_btn = False
                        uplaod_files = False


st.image("images/logo.png", width=150)
st.header("PDF to Chat Bot")
st.write("This is a simple web app to convert PDFs to Chat Bot")

question = st.text_input("Ask me a question about your data")
if question:
    if llm is None or db is None:
        st.warning("Please provide OpenAI API Key and Huggingface access token")
    else:
        answer, metadatas = llm.answer(question)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Metadata")
        st.write(metadatas)
