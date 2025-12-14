# The Simplest RAG: Private Files Q&A Bot

**A Containerized Retrieval-Augmented Generation (RAG) System using AWS Bedrock and LangChain.**

This project implements a foundational RAG pipeline that allows users to query a private set of documents (PDFs, TXT) and receive factual, cited answers. It emphasizes best practices, including Docker containerization, secure credential handling, and the use of the latest AWS Bedrock foundational models.

### Technology Stack & Architecture

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | Python (`langchain-classic`) | Defines the RAG pipeline logic (Ingestion and Q&A). |
| **LLM Provider** | AWS Bedrock (Amazon Nova-Lite) | Generative Model for final answer formulation. |
| **Embedding Model** | AWS Bedrock (Amazon Titan) | Converts private documents into numerical vectors. |
| **Vector Store** | Chroma DB | Stores and retrieves document embeddings for context. |
| **Containerization** | Docker | Provides an isolated and reproducible environment. |
| **Observability** | LangSmith | Tracing, debugging, and evaluating the RAG chain. |




### Quick Start Guide: Necessary Steps

These steps compile the complete workflow from setup to execution.

#### A. Prerequisites & Setup

1.  **Install Docker:** Ensure Docker is installed and running on your system.
2.  **AWS Bedrock Access:** Verify in the AWS console that **Amazon Titan Text Embeddings V1** and **Amazon Nova-Lite** are explicitly granted access for your AWS account in the target region (`us-east-1` by default).
3.  **Create `.env` File:** Create the file **`.env`** in the project root and add your credentials. **(IMPORTANT: Add `.env` to your `.gitignore`.)**

    ```dotenv
    # .env
    AWS_REGION=us-east-1
    AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID_HERE
    AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY_HERE
    
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=YOUR_LANGSMITH_API_KEY_HERE
    LANGCHAIN_PROJECT=private-qa-bot-bedrock
    ```


#### 4. Build the Docker Image

This command reads your `Dockerfile` and `requirements.txt` to create the container image.

docker build -t rag-bedrock-capstone:v1 .

Be sure that the document is already uploaded to /private_docs folder.

#### 5. Run the Ingestion
docker run --rm \       
  --env-file ./.env \
  -v "$(pwd)/private_qa_db:/app/private_qa_db" \
  rag-bedrock-capstone:v1 \
  ingest.py

#### 6. Run the Qna Bot
docker run -it --rm \
  --env-file ./.env \
  -v "$(pwd)/private_qa_db:/app/private_qa_db" \
  rag-bedrock-capstone:v1 \
  qa_bot.py


