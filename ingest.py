# ingest.py (REVISED FOR AWS BEDROCK)

import os
from langchain_classic.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, GoogleDriveLoader
from langchain_aws import BedrockEmbeddings # *** NEW IMPORT ***
import boto3 # *** NEW IMPORT ***

# --- Configuration ---
DOCS_DIRECTORY = "private_docs"
CHROMA_PATH = "private_qa_db" 
# BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1") # Read from ENV or default

def create_vector_database():
    """Loads documents, splits them, and stores them in Chroma using Bedrock Embeddings."""
    print("--- Starting Document Ingestion (Using AWS Bedrock) ---")
    
    # Check for AWS Credentials
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("ERROR: AWS_ACCESS_KEY_ID environment variable not set. Please configure AWS credentials.")
        return
    
    # 1. Load All Documents (Same as before)
    loaders = []
    for root, _, files in os.walk(DOCS_DIRECTORY):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loaders.append(PyPDFLoader(file_path))
            elif file.endswith(".txt"):
                loaders.append(TextLoader(file_path))
    
    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {loader.file_path}: {e}")
            
    if not documents:
        print("\nERROR: No supported documents loaded. Check your 'private_docs' folder (.pdf or .txt).")
        return

    print(f"Loaded {len(documents)} document(s) in total.")

    # 2. Split the Document (Same as before)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")

    # 3. Create Bedrock Embeddings & Store
    # The client is created automatically by LangChain using the ENV variables
    embeddings = BedrockEmbeddings(
        # region_name=BEDROCK_REGION, 
        # model_id="amazon.titan-embed-text-v2:0"  # Titan is a highly stable Bedrock embedding model
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=CHROMA_PATH
    )
    vector_db.persist()
    print(f"Successfully created and saved Chroma DB to '{CHROMA_PATH}' using Titan Embeddings.")
    print("--- Ingestion Complete ---")

if __name__ == "__main__":
    create_vector_database()