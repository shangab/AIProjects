from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from env import HUGGINGFACE_API_KEY
from datatext import data
import streamlit as st
import os


class AIHelper:
    def __init__(self):
        self.vector_db = None
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name="hkunlp/instructor-base",
            api_key=HUGGINGFACE_API_KEY)
        self.create_or_load_vector_db()

    def create_or_load_vector_db(self):
        texts = [x["category"]+": "+x["content"] for x in data]
        if not os.path.exists("vector_db"):
            self.vector_db = FAISS.from_texts(
                texts=texts, embedding=self.embeddings)
            FAISS.save_local(self.vector_db, "vector_db")
        else:
            self.vector_db = FAISS.load_local(
                folder_path="vector_db", embeddings=self.embeddings, allow_dangerous_deserialization=True)

    def embed_text(self, text):
        query_embedding = self.embeddings.embed_query(text)
        return query_embedding

    def search(self, query: str):
        if self.vector_db is None:
            return []
        return self.vector_db.similarity_search(query, k=2)

    def get_vectors_number(self):
        if self.vector_db is not None:
            return len(self.vector_db.index_to_docstore_id)
        return 0


ai = AIHelper()


st.title("🧠 Sematnic Search using Huggingface, FAISS and Langchain")
st.subheader(
    "Search for similar text in the dataset using the meaning not the keywords")
count = ai.get_vectors_number()
st.write(f"The vector database contains : {count}")

query = st.text_input("Enter your search query")
if query:
    results = ai.search(query)
    st.write(results)
    st.subheader("Query as Embedding")
    st.write(ai.embed_text(query))
