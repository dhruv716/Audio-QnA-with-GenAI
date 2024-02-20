from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import retriever
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_API_KEY"

def create_vector_store(file_path):
    
    with open(file_path) as f:
        state_of_the_union = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350,chunk_overlap=20)
    chunks = text_splitter.split_text(state_of_the_union)
    
    embeddings = HuggingFaceEmbeddings()
    
    vectorStore = FAISS.from_texts(chunks, embeddings)
    
    return vectorStore

def get_result(query):
    llm=HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'], repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.1 ,"max_length":512})
    vector_store = create_vector_store("/Users/dhruvpai/Downloads/WhisperAI/Version2/knowledge.txt")
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    result = chain.run(query)
    
    return result

def main():
    query = "What are the challenges of Decision Trees?"
    result = get_result(query)
    print(result)
    
main()
    


