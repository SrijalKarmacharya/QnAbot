# qa_bot.py - FINAL REVISED VERSION (Bedrock & LCEL Chains)

import os
import sys 
from langchain_chroma import Chroma
# --- NEW/UPDATED IMPORTS ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


from langchain_aws import BedrockEmbeddings, ChatBedrock
import boto3

# --- Configuration ---
CHROMA_PATH = "private_qa_db"


def run_private_qa_bot():
    """Loads the database and runs the RAG chain with Bedrock models using LCEL."""
    print("--- Starting Private Q&A Bot (Using Amazon Nova-Lite & LCEL) ---")
    
    # Check for AWS Configuration
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        print("ERROR: AWS_ACCESS_KEY_ID environment variable not set. Cannot run Bedrock models.")
        sys.exit(1)

    # 1. Load Embeddings and Vector Store
    # Embeddings model must match the one used during ingestion
    embeddings = BedrockEmbeddings(
        # region_name=BEDROCK_REGION, 
        # model_id="amazon.titan-embed-text-v2:0"
    )
    
    try:
        vector_db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=CHROMA_PATH

        )
    except Exception as e:
        print(f"Error loading database: {e}")
        print("Please run 'python ingest.py' first.")
        sys.exit(1)

    # 2. Setup Retriever and LLM
    retriever = vector_db.similarity_search(k=9)

    # Bedrock LLM: Claude 3 Haiku via Converse API
    llm = ChatBedrock(
        # region_name=BEDROCK_REGION, 
        model_id="amazon.nova-lite-v1:0", 
        temperature=0.0
    )

    # 3. Define the PDO Prompt Template (Using ChatPromptTemplate for Message API compatibility)
    
    # P (Persona) & O (Output Instruction) are placed in the System Message
    SYSTEM_PROMPT = """
    You are a meticulous research assistant. Your primary function is to answer the user's question FACTUALLY based ONLY on the context provided below. 
    If you cannot find the answer in the context, you MUST state, "I apologize, the information required is not found in the documents."
    
    TASK: Answer the question directly and precisely. After your answer, provide the citation by appending: (Source: [file_name] - page [page_number]). 
    """
    
    # D (Data) is included in the User Message alongside the question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "---CONTEXT---\n{context}\n---END CONTEXT---\n\nQuestion: {input}"),
        ]
    )

    # 4. Create the LangChain RAG Chain (Using LCEL components)

    # a. Document Chain: Combines the retrieved documents with the prompt
    # This component correctly formats the documents and prompt into the 'messages' structure
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    # b. Retrieval Chain: Puts everything together (Retriever -> Document Chain)
    # The create_retrieval_chain handles mapping the query input to the retriever
    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    # 5. User Interaction Loop
    print("\nSystem Ready (using Amazon Nova-Lite). Ask a question, or type 'quit' to exit.")
    
    while True:
        query = input("\nYour Question: ")
        if query.lower() in ["quit", "exit"]:
            break
        
        print("Searching documents...")
        
        # Invoke the NEW Chain structure. The input key is simply "input" 
        # because create_retrieval_chain handles the mapping.
        result = qa_chain.invoke({
            "input": query
        })

        print("\n--- Bot Response ---")
        # The output key is now 'answer' instead of 'result'
        print(result['answer'])
        print("--------------------")

if __name__ == "__main__":
    run_private_qa_bot()