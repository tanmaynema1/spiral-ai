# Spiral AI - Research Document Chatbot

Spiral AI is an end-to-end chatbot application that empowers users to upload their Research documents and utilize the platform to ask questions, seek information, and gain insights regarding their documents. It combines cutting-edge technologies to enhance document understanding and retrieval, making it a valuable tool for Research professionals, researchers, and anyone dealing with Research documents.

## Features

- **Document Extraction**: Spiral AI extracts text content from uploaded Research documents, making them ready for analysis.

- **Text Chunking**: The content is divided into manageable chunks with significant overlap to ensure seamless connectivity between the produced document segments.

- **Embeddings**: Utilizes FAISS and OpenAI embeddings to convert text chunks into three-dimensional vectors. This embedding process enhances response time and system performance.

- **Storage**: The embedded vectors are stored for efficient retrieval and quick access, ensuring that users receive prompt responses to their queries.

- **OpenAI Language Model (LLM)**: Users can query the document using OpenAI's Language Model (LLM), enabling natural language interaction with the Research document.

## Getting Started

To get started with Spiral AI, follow these steps:

1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Install the necessary dependencies by running `pip install -r requirements.txt`.

3. **Add a .env File**: Create a new .env file and add your OPENAI_API_KEY.
   
4. **Run the Application**: Launch the Spiral AI application and start uploading your Research documents. 

5. **Ask Questions**: Once your document is uploaded, use the chat interface to ask questions and receive insights.

## Usage

1. Upload your Research document.

2. Start a conversation by asking questions about the document's content.

3. Spiral AI will utilize its text extraction, chunking, and embeddings capabilities to provide relevant responses.

## Technologies Used

- [Langchain](https://streamlit.io/)
- [OpenAI](https://python.langchain.com)
- [Streamlit](https://platform.openai.com/docs/models)

## About Me

Spiral AI is created by [Tanmay Nema](https://www.linkedin.com/in/tanmay-nema-0754721bb/).
