import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Securely load OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Function to process PDFs and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    if not text.strip():
        st.warning("No readable text was found in the uploaded PDF.")
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

# Create FAISS vectorstore from text chunks
def get_vectorstore(text_chunks):
    if not text_chunks:
        st.error("No text chunks available for processing. Please upload a valid PDF.")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Get conversational chain for the sales coach functionality
def get_conversation_chain(vectorstore, selected_model):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(api_key=openai_api_key, model_name=selected_model),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

# Handle user input
def handle_userinput(user_question, vectorstore):
    if not st.session_state.conversation:
        st.error("Please upload a PDF and add data before asking questions.")
        return
    response = st.session_state.conversation({"question": user_question})
    if response and "answer" in response:
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
        if st.session_state.current_session:
            st.session_state.sessions[st.session_state.current_session] = st.session_state.chat_history
    else:
        st.error("Failed to get a response from GPT.")

# Main Streamlit application
def main():
    st.set_page_config(page_title="Chat with Sales Coach 🚗", page_icon="🚗")

    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-3.5-turbo"

    st.header("Chat with Sales Coach 🚗")

    # Add CSS for chat layout
    st.markdown("""
        <style>
        .user-message {
            background-color: #FFFFFF; /* White background for user messages */
            padding: 8px 12px;
            border-radius: 12px;
            text-align: left;
            margin-left: auto; /* Push the message to the right */
            margin-right: 10px;
            margin-bottom: 10px; /* Add space below user message */
            max-width: 70%;
            color: #000; /* Black text color */
            display: block;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        }
        .assistant-message {
            background-color: #D6EAF8; /* Light Blue background for assistant messages */
            padding: 8px 12px;
            border-radius: 12px;
            text-align: left;
            margin-left: 10px; /* Push the message to the left */
            margin-right: auto;
            margin-bottom: 15px; /* Add space below assistant message */
            max-width: 70%;
            color: #000; /* Black text color */
            display: block;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px; /* Space between messages in the container */
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for sessions
    with st.sidebar:
        st.subheader("Chat Sessions")
        available_models = ["gpt-3.5-turbo", "gpt-4"]
        selected_model = st.selectbox(
            "Select GPT Model",
            available_models,
            index=0,
            key="model_selection"
        )
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            new_session_name = f"{selected_model} - Chat {len(st.session_state.sessions) + 1}"
            st.session_state.sessions[new_session_name] = []
            st.session_state.current_session = new_session_name
            st.session_state.chat_history = []

        for session_name in list(st.session_state.sessions.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(session_name):
                    st.session_state.current_session = session_name
                    st.session_state.chat_history = st.session_state.sessions[session_name].copy()
            with col2:
                if st.button("❌", key=f"delete_{session_name}"):
                    del st.session_state.sessions[session_name]
                    if session_name == st.session_state.current_session:
                        st.session_state.current_session = None
                        st.session_state.chat_history = []
                    st.rerun()

        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Add Data'", accept_multiple_files=True)
        if st.button("Add Data"):
            with st.spinner("Processing PDFs..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable content found in the uploaded PDFs.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        if vectorstore:
                            st.session_state.conversation = get_conversation_chain(vectorstore, st.session_state.selected_model)
                        st.success("PDFs have been processed successfully!")
                except Exception as e:
                    st.error(f"Failed to process PDFs: {e}")

    # Input box for user's question
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Ask your question:",
            value=st.session_state.user_input,
            key="dynamic_user_input"
        )
    with col2:
        send_button = st.button("Send")

    if send_button:
        if user_input.strip():
            st.session_state.current_question = user_input.strip()
            st.session_state.chat_history.append({"role": "user", "content": st.session_state.current_question})
            handle_userinput(st.session_state.current_question, st.session_state.conversation)
            st.session_state.user_input = ""
            st.rerun()
        else:
            st.warning("Please enter a valid question.")

    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Start by asking a question!")

if __name__ == "__main__":
    main()
