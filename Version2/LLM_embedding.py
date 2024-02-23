import streamlit as st
import chromadb
import uuid
import chromadb.utils.embedding_functions as embedding_functions
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import BasePromptTemplate
from langchain_community.llms import HuggingFaceTextGenInference
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import SystemMessage
from openai import OpenAI




os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
client = OpenAI()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai_api_key = "YOUR_API_KEY"


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def compute_similarity(embedding1, embedding2):
    
    dot_product = np.dot(embedding1, embedding2)
    norm_product = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)

    return dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

class CustomSystemMessagePromptTemplate(FewShotPromptTemplate):
    def format(self, *args, **kwargs):
        # existing implementation here
        pass

    def format_prompt(self, *args, **kwargs):
        # implementation here
        pass

class Runnable:
    def __init__(self, type, model_name, max_new_tokens=512, top_k=50, temperature=0.1, repetition_penalty=1.03):
        self.type = type
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

def load_chroma_vector_store():
    try:
        print("Getting the Chroma Client...")
        chroma_client = chromadb.Client()

        collection_name = f"collection_{uuid.uuid4()}"

        print("Checking for the existence of Chroma Collection...")
        existing_collections = chroma_client.list_collections()

        if collection_name not in existing_collections:
            print(f"Creating a Chroma Collection '{collection_name}'...")
            metadata = {"description": "Chroma Vector Store for audio transcripts"}
            collection = chroma_client.create_collection(name=collection_name, metadata=metadata)

            loader = TextLoader("knowledge.txt")
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            
            embedding_functions = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

            collection.add(
                documents=[doc.page_content for doc in docs],
                metadatas=[{"source": f"Document_{i}"} for i in range(len(docs))],
                ids=[f"Document_{i}" for i in range(len(docs))]
            )

            print("Chroma Vector Store loaded successfully.")
        else:
            print(f"Chroma Vector Store with the name '{collection_name}' already exists.")
            collection = chroma_client.get_collection(collection_name)

        return collection

    except Exception as e:
        print(f"Error loading Chroma Vector Store: {e}")
        return None

def create_chat_prompt_template():
    template_messages = [
        SystemMessage(content="You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    return ChatPromptTemplate.from_messages(template_messages)

def llama_langchain_chroma_integration(csv_file_path, chroma_vector_store, openai_api_key):
    try:
        st.info("Initializing LangChain LLMChain...")

        llm_chain = LLMChain(
            prompt=create_chat_prompt_template(),
            llm=RunnablePassthrough(type="huggingface", model_name="textgen-ada-002"),
        )
        with open(csv_file_path, "r") as csv_file:
            transcript_text = csv_file.read()

        st.header("Querying LangChain-Chroma-Llama Integration:")
        st.subheader("Transcription Text:")
        st.write(transcript_text)

        st.subheader("OpenAI Embedding Search Results:")
        
        query_embedding = get_embedding(transcript_text, model="text-embedding-3-small")

        chroma_vector_store = load_chroma_vector_store()
        if chroma_vector_store is None:
            st.error("Chroma Vector Store is not loaded. Aborting integration.")
            return

        document_embeddings = []
        document_contents = []
        for document in chroma_vector_store:
            page_content = document.page_content
            embedding = get_embedding(page_content, model="text-embedding-3-small")
            document_embeddings.append(embedding)
            document_contents.append(page_content)
            
        scores = [compute_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]

        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        top_5_indices = sorted_indices[:5]

        for i in range(5):
            st.write(f"Similarity Score: {scores[top_5_indices[i]]}")
            st.write(f"Document Content: {document_contents[top_5_indices[i]]}")

        st.subheader("LangChain-Llama Search Results:")
        model = Llama2Chat(llm_chain=llm_chain)

        chat_prompt_template = create_chat_prompt_template()

        chat_prompt = chat_prompt_template.format(chat_history=[], text=transcript_text)

        response = model.run(prompt=chat_prompt)
        for message in response:
            st.write(f"Chat Message: {message}")

        return response

    except Exception as e:
        st.error(f"Error in llama_langchain_chroma_integration: {e}")

def main():
    st.title("LangChain-Chroma-Llama Integration")
    openai_api_key = "YOUR_API_KEY"

    chroma_vector_store = load_chroma_vector_store()
    print(chroma_vector_store)

    if chroma_vector_store:
        csv_file_path = "transcription_output.csv"
        llama_langchain_chroma_integration(csv_file_path, chroma_vector_store, openai_api_key)
