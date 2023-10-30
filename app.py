from dotenv import load_dotenv
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os

from langchain.text_splitter import CharacterTextSplitter

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

    # create collection - run once
    # os.environ['QDRANT_COLLECTION'] = os.getenv("QDRANT_COLLECTION_NAME")

    # collection_config = qdrant_client.http.models.VectorParams(
    #         size=1536, # 768 for instructor-xl, 1536 for OpenAI
    #         distance=qdrant_client.http.models.Distance.COSINE
    #     )

    # client.recreate_collection(
    #     collection_name=os.getenv("QDRANT_COLLECTION"),
    #     vectors_config=collection_config
    # )

    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store

def main():
    load_dotenv()
    
    st.set_page_config(page_title="Ask Qdrant")
    st.header("Ask your remote database 💬")

    
    
    # create vector store
    vector_store = get_vector_store()

    with open("story.txt") as f:
        raw_text = f.read()

    texts = get_chunks(raw_text)

    vector_store.add_texts(texts)
    
    # create chain 
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # show user input
    user_question = st.text_input("Ask a question about your database:")
    if user_question:
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")
    
        
if __name__ == '__main__':
    main()
