import os
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings

from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.chains.llm import LLMChain
import streamlit as st


class VectorDatabase:
    def __init__(self) -> None:
        self.vectordb = None
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name="hkunlp/instructor-base", api_key=st.secrets["HUGGINGFACE_ACCESS_TOKEN"]) if st.secrets["HUGGINGFACE_ACCESS_TOKEN"] is not None else None

        if os.path.exists("vector_db"):
            self.vectordb = FAISS.load_local(folder_path="vector_db",
                                             embeddings=self.embeddings,
                                             allow_dangerous_deserialization=True)

    def update_db(self, texts=[str], metadatas=[dict]):
        if self.vectordb is None:
            self.vectordb = FAISS.from_texts(
                texts=texts, embedding=self.embeddings, metadatas=metadatas)
        else:
            self.vectordb.add_texts(texts=texts, metadatas=metadatas)

        self.vectordb.save_local(folder_path="vector_db")

    def search(self, query):
        if self.vectordb is None:
            return []

        return self.vectordb.similarity_search(query=query, k=3)

    def get_chunks_count(self):
        if self.vectordb is None:
            return 0

        return len(self.vectordb.index_to_docstore_id)


class LLMHelper:
    def __init__(self) -> None:
        self.llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"],
                          temperature=0, max_tokens=700) if st.secrets["OPENAI_API_KEY"] is not None else None
        self.prompt = PromptTemplate(
            template="""
                    You are assitant to answer questions from a context given to you only. 
                    If the context is empty please apologize and say I do not know!!!
                    Context: {context}
                    Question: {question}

                """,
            input_variables=["context", "question"]

        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.db = VectorDatabase()

    def answer(self, question):
        context = ""
        chunks = self.db.search(query=question)
        footer = []
        for chunk in chunks:
            context += chunk.page_content
            footer.append(chunk.metadata)
        response = self.chain.run(context=context, question=question)

        return response, footer
