import os
import json
import requests
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings import SentenceTransformerEmbeddings
from typing import Any, List, Mapping, Optional
import google.generativeai as genai
import time


# Custom LLM class for local Llama models (assuming Ollama)
class LocalLlama(LLM):
    """Custom LLM wrapper for local Llama models running via Ollama."""
    
    # You can change this URL to match your local setup
    api_url: str = "http://localhost:11434/api/generate"
    model_name: str = "llama3"
    temperature: float = 0.7
    max_tokens: int = 1024
    
    @property
    def _llm_type(self) -> str:
        return "local_llama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the local Llama API."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            st.error(f"Error calling local Llama model: {str(e)}")
            return f"Error: {str(e)}"


# Custom LLM class for Google Gemini
class GeminiLLM(LLM):
    """Custom LLM wrapper for Google Gemini."""
    
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_tokens: int = 1024
    
    def __init__(self, api_key=None, **kwargs):
        super().__init__(**kwargs)
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key is required")
        genai.configure(api_key=api_key)
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Gemini API."""
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error calling Gemini model: {str(e)}")
            return f"Error: {str(e)}"


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split text into chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def create_knowledge_base(chunks):
    """Create a vector store from text chunks."""
    try:
        # Use SentenceTransformer embeddings directly without Hugging Face
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        return knowledge_base
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None


def get_llm(model_choice, llama_version=None, temperature=0.7):
    """Get the selected language model."""
    try:
        if "Llama" in model_choice:
            # For local Llama models
            llm = LocalLlama(temperature=temperature)
            if llama_version:
                llm.model_name = llama_version
            return llm
        elif model_choice == "Google Gemini":
            # For Google Gemini
            return GeminiLLM(temperature=temperature)
        else:
            st.error(f"Unknown model choice: {model_choice}")
            return None
    except Exception as e:
        st.error(f"Error initializing language model: {str(e)}")
        return None


def main():
    # Load environment variables
    load_dotenv()
    
    # Page configuration
    st.set_page_config(
        page_title="Ask your PDF",
        page_icon="📚",
        layout="wide"
    )
    
    # App header
    st.header("📚 Ask your PDF 💬")
    st.markdown("""
    Upload a PDF document and ask questions about its content.
    This app uses your local Llama models or Google's Gemini to answer questions.
    """)
    
    # Sidebar with information and settings
    with st.sidebar:
        st.subheader("Model Settings")
        
        # Model selection
        model_choice = st.radio(
            "Select Model Family",
            ["Local Llama", "Google Gemini"],
            index=0
        )
        
        # Llama version selection
        llama_version = None
        if model_choice == "Local Llama":
            llama_version = st.selectbox(
                "Select Llama Version",
                ["llama3", "llama3.1", "llama2"],
                index=0,
                help="Choose which Llama model version to use"
            )
        
        # Temperature setting
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        # If Gemini is selected, check for API key
        if model_choice == "Google Gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                google_api_key = st.text_input("Enter Google API Key:", type="password")
                if google_api_key:
                    os.environ["GOOGLE_API_KEY"] = google_api_key
                    st.success("Google API Key set!")
        
        # Local Llama settings (if needed)
        if model_choice == "Local Llama":
            st.subheader("Local Llama Settings")
            llama_url = st.text_input(
                "Llama API URL:",
                value="http://localhost:11434/api/generate",
                help="URL for your local Llama API"
            )
            
            # Display model setup instructions
            with st.expander("Llama Model Setup Instructions"):
                st.markdown("""
                ### Setting up Llama models with Ollama
                
                If you're using Ollama, you can pull the models with these commands:
                
                ```bash
                # For Llama 3
                ollama pull llama3
                
                # For Llama 3.1
                ollama pull llama3.1
                
                # For Llama 2
                ollama pull llama2
                ```
                
                Make sure the model name in the dropdown matches exactly what you pulled in Ollama.
                """)
        
        st.subheader("About")
        st.markdown("""
        This application allows you to:
        - Upload a PDF document
        - Process it using NLP techniques
        - Ask questions about the content
        
        The app uses:
        - SentenceTransformer embeddings
        - FAISS for vector storage
        - Your choice of LLM for question answering
        """)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Upload file
        st.subheader("Upload Document")
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        
        # Process button
        process_button = st.button("Process Document", disabled=pdf is None)
    
    # Session state to store the knowledge base
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    
    # Process PDF when button is clicked
    if pdf is not None and process_button:
        with st.spinner("Processing PDF..."):
            # Extract text
            text = extract_text_from_pdf(pdf)
            st.success(f"PDF processed: {len(text)} characters extracted")
            
            # Create text chunks
            chunks = get_text_chunks(text)
            st.success(f"Text split into {len(chunks)} chunks")
            
            # Create knowledge base
            with st.spinner("Creating knowledge base (this may take a moment)..."):
                st.session_state.knowledge_base = create_knowledge_base(chunks)
            
            if st.session_state.knowledge_base:
                st.success("Knowledge base created successfully!")
    
    with col2:
        # Show user input if knowledge base exists
        if st.session_state.knowledge_base:
            st.subheader("Ask Questions")
            user_question = st.text_input("What would you like to know about the document?")
            
            if user_question:
                # Update LocalLlama settings if needed
                if model_choice == "Local Llama":
                    LocalLlama.api_url = llama_url
                
                # Get the language model
                llm = get_llm(model_choice, llama_version, temperature)
                
                if llm:
                    model_display_name = f"{model_choice} ({llama_version})" if llama_version else model_choice
                    with st.spinner(f"Thinking using {model_display_name}..."):
                        try:
                            # Get relevant documents
                            start_time = time.time()
                            docs = st.session_state.knowledge_base.similarity_search(user_question, k=4)
                            search_time = time.time() - start_time
                            
                            # Create and run the chain
                            start_time = time.time()
                            chain = load_qa_chain(llm, chain_type="stuff")
                            response = chain.run(input_documents=docs, question=user_question)
                            answer_time = time.time() - start_time
                            
                            # Display the answer
                            st.markdown("### Answer:")
                            st.markdown(response)
                            
                            # Display performance metrics
                            st.markdown("---")
                            st.markdown(f"**Model used:** {model_display_name}")
                            st.markdown(f"**Search time:** {search_time:.2f}s | **Answer generation time:** {answer_time:.2f}s")
                            
                            # Display source documents
                            with st.expander("View source documents"):
                                for i, doc in enumerate(docs):
                                    st.markdown(f"**Document {i+1}**")
                                    st.markdown(doc.page_content)
                                    st.markdown("---")
                            
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                            
                            if model_choice == "Local Llama":
                                st.info(f"Make sure your local {llama_version} model is installed and the Ollama server is running.")
                                st.code(f"ollama pull {llama_version}", language="bash")
                            elif model_choice == "Google Gemini":
                                st.info("Make sure your Google API key is valid and has access to Gemini.")
        else:
            st.info("👈 Upload and process a PDF document to start asking questions")


if __name__ == '__main__':
    main()