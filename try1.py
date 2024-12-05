import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Changed from FAISS
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import json
import warnings
import time
# Add after imports
import torch
from tqdm.auto import tqdm

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
else:
    print("No GPU available, using CPU")
    DEVICE = "cpu"
# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class ChatGroqManager:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.model_name = "llama-3.2-90b-vision-preview"  # Using the working model

    def create_llm(self, temperature=0.4):
        return ChatGroq(
            temperature=temperature,
            groq_api_key=self.groq_api_key,
            model_name=self.model_name
        )

def test_llm():
    try:
        groq_manager = ChatGroqManager()
        llm = groq_manager.create_llm()
        test_response = llm.invoke("Test connection - what is 2+2? Reply with just the number.")
        print("LLM Connection Test Successful!")
        return True
    except Exception as e:
        st.error(f"LLM Connection Error: {str(e)}")
        return False

class PharmaKnowledgeAssistant:
    def __init__(self):
        with st.status("Initializing Assistant...", expanded=True) as status:
            try:
                # LLM Initialization
                print("Starting LLM initialization...")
                status.write("🔄 Initializing LLM connection...")
                self._initialize_llm()
                status.write("✅ LLM initialized successfully!")
                print("LLM initialization complete")
                
                # Embeddings Initialization
                print("Starting embeddings initialization...")
                status.write("🔄 Setting up embeddings model...")
                self._initialize_embeddings()
                status.write("✅ Embeddings model loaded!")
                print("Embeddings initialization complete")
                
                # Vectorstore Initialization
                print("Starting vectorstore initialization...")
                status.write("🔄 Loading knowledge base...")
                status.write("📚 Reading JSON files...")
                self._initialize_vectorstore()
                status.write("✅ Knowledge base loaded and indexed!")
                print("Vectorstore initialization complete")
                
                # Tools Setup
                print("Starting tools initialization...")
                status.write("🔄 Setting up QA tools...")
                self._initialize_tools()
                status.write("✅ Tools configured!")
                print("Tools initialization complete")
                
                # Agent Setup
                print("Starting agent initialization...")
                status.write("🔄 Finalizing agent setup...")
                self._initialize_agent()
                status.write("✅ Agent ready!")
                print("Agent initialization complete")
                
                status.update(label="✨ Assistant Ready!", state="complete")
            except Exception as e:
                print(f"Error during initialization: {str(e)}")
                status.update(label=f"❌ Error: {str(e)}", state="error")
                raise

    def _initialize_llm(self):
        self.groq_manager = ChatGroqManager()
        self.llm = self.groq_manager.create_llm()

    def _initialize_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='all-MiniLM-L6-v2',
            model_kwargs={'device': DEVICE},  # Use GPU if available
            encode_kwargs={'device': DEVICE, 'batch_size': 32}  # Batch processing on GPU
        )

    def _initialize_vectorstore(self):
        print("Checking for existing ChromaDB database...")
        import chromadb
        from chromadb.config import Settings
        
        # Initialize ChromaDB client
        chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory="chroma_db"
        ))
        
        # Check if collection exists
        try:
            existing_collection = chroma_client.get_collection(name="pharma_knowledge_base")
            if existing_collection and existing_collection.count() > 0:
                print("Found existing ChromaDB collection with embeddings")
                self.vectorstore = Chroma(
                    client=chroma_client,
                    collection_name="pharma_knowledge_base",
                    embedding_function=self.embeddings
                )
                print(f"Loaded existing database with {existing_collection.count()} entries")
                return
        except ValueError:
            print("No existing collection found, creating new embeddings...")
        
        # If no existing collection, create new embeddings
        print("Starting to read JSON files...")
        docs = []
        data_dir = "microlabs_usa"
        file_count = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
        print(f"Found {file_count} JSON files to process")
        
        # Use tqdm for progress bar
        for filename in tqdm(os.listdir(data_dir), desc="Loading files"):
            if filename.endswith(".json"):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = json.dumps(data, indent=2)
                    docs.append(text)

        print("Starting text splitting...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.create_documents(docs)
        total_chunks = len(splits)
        print(f"Created {total_chunks} text chunks")
        
        print("Creating new ChromaDB database...")
        
        # Process in smaller batches optimized for GPU
        batch_size = 64  # Optimized for RTX 3050 Ti
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        print(f"Processing embeddings in {total_batches} batches on {DEVICE}...")
        
        # Show progress bar for embedding creation
        with tqdm(total=total_chunks, desc="Creating embeddings") as pbar:
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                
                # Create or update vectorstore
                if i == 0:
                    self.vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        client=chroma_client,
                        collection_name="pharma_knowledge_base"
                    )
                else:
                    self.vectorstore.add_documents(documents=batch)
                
                pbar.update(len(batch))
                
                # Optional: Clear CUDA cache periodically
                if DEVICE == "cuda" and i % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
        
        print("ChromaDB database created successfully")






            
    def _initialize_tools(self):
        # Wrapper function to handle the RetrievalQA output format
        def qa_wrapper(query: str) -> str:
            result = self.qa(query)
            # Extract just the result text, ignoring source documents
            return result['result'] if isinstance(result, dict) else result

        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

        self.tools = [
            Tool(
                name="Product_Knowledge_Base",
                func=qa_wrapper,  # Use the wrapper function instead of self.qa.run
                description="Use this tool to answer questions about pharmaceutical products."
            ),
            Tool(
                name="Summarizer",
                func=self.summarize,
                description="Use this tool to generate summaries of pharmaceutical products."
            ),
            Tool(
                name="Recommender",
                func=self.recommend,
                description="Use this tool to provide recommendations based on symptoms or conditions."
            )
        ]
    def _initialize_agent(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def summarize(self, product_name: str) -> str:
        context = self.vectorstore.similarity_search(product_name, k=4)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a concise summary of the pharmaceutical product."),
            ("human", "{product_name}\n\nContext: {context}")
        ])
        chain = prompt | self.llm
        return chain.invoke({
            "product_name": product_name,
            "context": "\n".join([doc.page_content for doc in context])
        })

    def recommend(self, query: str) -> str:
        context = self.vectorstore.similarity_search(query, k=4)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide recommendations based on the pharmaceutical knowledge base."),
            ("human", "{query}\n\nContext: {context}")
        ])
        chain = prompt | self.llm
        return chain.invoke({
            "query": query,
            "context": "\n".join([doc.page_content for doc in context])
        })

def main():
    # Page configuration must come first
    st.set_page_config(
        page_title="Pharmaceutical Knowledge Assistant",
        page_icon="💊",
        layout="wide"
    )

    # Custom CSS with enhanced animations and styling
    st.markdown("""
        <style>
        /* Main background with enhanced radial gradient */
        .stApp {
            background: radial-gradient(circle at center, #1e2a4a 0%, #0a0a1a 100%);
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Header styling with animation */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 2rem;
            animation: fadeInDown 0.8s ease-out;
        }
        
        /* Header icons styling */
        .header-icon {
            font-size: 2rem;
            margin: 0 0.5rem;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
        
        /* Chat message container styling with animations */
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 15px !important;
            padding: 1rem !important;
            margin: 1rem 0 !important;
            backdrop-filter: blur(10px) !important;
            transform: translateX(0);
            transition: all 0.3s ease-out !important;
            animation: slideIn 0.5s ease-out;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        .stChatMessage:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2) !important;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* User message styling with glow effect */
        .stChatMessage.user {
            background-color: rgba(70, 100, 200, 0.1) !important;
            border: 1px solid rgba(100, 150, 255, 0.2) !important;
            animation: slideInRight 0.5s ease-out;
        }
        
        .stChatMessage.user:hover {
            box-shadow: 0 0 15px rgba(100, 150, 255, 0.1) !important;
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Assistant message styling with subtle gradient */
        .stChatMessage.assistant {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.05)) !important;
            animation: slideInLeft 0.5s ease-out;
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Input box styling with glow effect */
        .stTextInput input {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 0.8rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput input:focus {
            border-color: rgba(100, 150, 255, 0.5) !important;
            box-shadow: 0 0 20px rgba(100, 150, 255, 0.2) !important;
            transform: translateY(-1px);
        }
        
        /* Status indicator styling with pulse animation */
        .stStatus {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Spinner styling with glow */
        .stSpinner > div {
            border-color: #4664c8 !important;
            box-shadow: 0 0 15px rgba(70, 100, 200, 0.3) !important;
        }
        
        /* Text color and hover effects */
        .stMarkdown, .stText {
            color: #e6e6e6 !important;
            transition: color 0.3s ease !important;
        }
        
        .stMarkdown:hover, .stText:hover {
            color: #ffffff !important;
        }
        
        /* Error message styling with fade */
        .stError {
            background-color: rgba(255, 80, 80, 0.1) !important;
            border: 1px solid rgba(255, 80, 80, 0.2) !important;
            color: #ff8080 !important;
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Custom scrollbar with glow */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
        }
        
        /* Fade in animation */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Enhanced header with animated icons
    st.markdown("""
        <div class="main-header">
            <span class="header-icon">💊</span>
            Pharmaceutical Knowledge Assistant
            <span class="header-icon">🔬</span>
        </div>
    """, unsafe_allow_html=True)

    print("Starting main application...")
    # Test LLM connection first
    with st.spinner("🔄 Testing LLM connection..."):
        print("Testing LLM connection...")
        if not test_llm():
            st.error("❌ Failed to connect to LLM. Please check your API key and internet connection.")
            st.stop()
            return
        print("LLM connection test successful")

    # Initialize the assistant
    if 'assistant' not in st.session_state:
        try:
            print("Starting assistant initialization...")
            st.session_state.assistant = PharmaKnowledgeAssistant()
            st.session_state.messages = []
            print("Assistant initialization completed successfully")
        except Exception as e:
            print(f"Failed to initialize assistant: {str(e)}")
            st.error(f"❌ Failed to initialize assistant: {str(e)}")
            st.stop()
            return

    # Create a container for the chat interface
    chat_container = st.container()
    
    # Display chat interface with enhanced styling
    with chat_container:
        for message in st.session_state.messages:
            icon = "🧑‍💼" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"]):
                st.markdown(f"{icon} {message['content']}")

    # Handle user input with enhanced styling
    if prompt := st.chat_input("💭 Ask me about pharmaceutical products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"🧑‍💼 {prompt}")

        with st.chat_message("assistant"):
            try:
                with st.spinner('🤔 Processing your question...'):
                    response = st.session_state.assistant.agent.run(prompt)
                st.markdown(f"🤖 {response}")
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()