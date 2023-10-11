import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.buy_me_a_coffee import button
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os


with st.sidebar:
    st.title("Spiral 🧬")
    st.markdown('''
    Join millions of students, researchers and professionals to
    instantly answer questions about your legal document and understand
    research with AI.
                
    This Application was built using:
    - [Langchain](https://streamlit.io/)
    - [OpenAI](https://python.langchain.com)
    - [Streamlit](https://platform.openai.com/docs/models)
    ''')
    
    add_vertical_space(1)

    pdf = st.sidebar.file_uploader("Upload your Legal Document", type='pdf')

    add_vertical_space(1)

    button(username="tanmaynema", floating=False, width=220, font="Poppins", text="Caffeine?")

    add_vertical_space(1)

    st.write("Made by [Tanmay Nema](https://www.linkedin.com/in/tanmay-nema-0754721bb/)")


def main():
    
    load_dotenv()
    
    st.header("💬 Chat with your Legal Document")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text = text)

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)

        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding = embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if query := st.chat_input(placeholder="Ask any Question about your Legal Document:"):
            st.session_state.messages.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            docs = VectorStore.similarity_search(query = query, k = 3)

            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type = "stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            # Add the assistant's response to the chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.chat_message("user").write(chat["content"])
        else:
            st.chat_message("assistant").write(chat["content"])


if __name__ == '__main__':
    main()