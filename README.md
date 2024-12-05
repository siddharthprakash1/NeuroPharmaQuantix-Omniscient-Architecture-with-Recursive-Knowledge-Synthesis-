# NeuroPharmaQuantix Omniscient Architecture with Recursive Knowledge Synthesis v2.7.3-Helios

<p align="center">
  <strong>🧬 Advanced Pharmaceutical Knowledge Assistant powered by Neural Embeddings and GroqAI 🤖</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-v3.8+-blue.svg">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-🦜-green">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-🎈-red">
  <img alt="GroqAI" src="https://img.shields.io/badge/GroqAI-⚡-purple">
</p>

## 🌟 Overview

NPQ-OARKS is a sophisticated pharmaceutical knowledge assistant that leverages neural embeddings and large language models to provide accurate, context-aware information about pharmaceutical products. Built using state-of-the-art technologies, it offers an intuitive chat interface for accessing detailed pharmaceutical knowledge.

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph Data Pipeline
        A[Web Scraper] -->|JSON Files| B[Data Processing]
        B -->|Text Chunks| C[Neural Embeddings]
        C -->|Vector Store| D[ChromaDB]
    end
    
    subgraph Core System
        E[GroqAI LLM] -->|Query Processing| F[Knowledge Synthesis]
        D -->|Context Retrieval| F
        F -->|Response Generation| G[Agent System]
    end
    
    subgraph User Interface
        H[Streamlit Frontend] -->|User Query| G
        G -->|Enhanced Response| H
    end
    
    style Data Pipeline fill:#1e2a4a,color:#fff
    style Core System fill:#2a1e4a,color:#fff
    style User Interface fill:#4a1e2a,color:#fff
```

## ✨ Features

- **Advanced Neural Processing**: Utilizes HuggingFace's all-MiniLM-L6-v2 for state-of-the-art text embeddings
- **Intelligent Context Synthesis**: Implements recursive character splitting for optimal text processing
- **GPU-Accelerated Performance**: Leverages CUDA acceleration for enhanced processing speed
- **Persistent Knowledge Store**: ChromaDB-based vector storage for efficient information retrieval
- **Interactive UI**: Sleek, animated interface with realtime response visualization
- **Multi-Tool Integration**: 
  - Product Knowledge Base
  - Smart Summarizer
  - Intelligent Recommender

## 🚀 Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/NeuroPharmaQuantix-OARKS.git
cd NeuroPharmaQuantix-OARKS
```

2. **Environment Setup**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

3. **Configure Environment Variables**
```bash
# Create .env file
GROQ_API_KEY=your_api_key_here
```

4. **Run the Application**
```bash
streamlit run try1.py
```

## 💻 Technical Implementation

### Data Pipeline
- **Web Scraping**: Custom scraper for pharmaceutical data acquisition
- **Text Processing**: RecursiveCharacterTextSplitter with 500-token chunks
- **Embedding Generation**: GPU-optimized batch processing
- **Vector Storage**: Persistent ChromaDB implementation

### Core Components
```python
# Key System Components
├── ChatGroqManager        # LLM Interface Management
├── PharmaKnowledgeAssistant
│   ├── Embeddings Engine
│   ├── Vector Store
│   ├── Tools System
│   └── Agent Framework
└── UI System
```

## 🛠️ Development Stack

- **Frontend**: Streamlit with custom CSS animations
- **Backend**: Python with LangChain framework
- **Database**: ChromaDB for vector storage
- **AI Models**: 
  - GroqAI llama-3.2-90b-vision-preview
  - HuggingFace all-MiniLM-L6-v2

## 🎯 Performance Optimization

- Batch processing for embedding generation
- CUDA acceleration for GPU systems
- Efficient memory management with periodic cache clearing
- Optimized chunk size for context retrieval

## 🎨 UI/UX Features

- Dark mode interface with radial gradient
- Animated message transitions
- Interactive chat bubbles
- Real-time response indicators
- Custom scrollbar implementation
- Responsive design elements

## 📊 System Architecture Components

```mermaid
graph TB
    A[Web Interface] -->|User Input| B{Router}
    B -->|Queries| C[LLM Processor]
    B -->|Searches| D[Vector Store]
    C -->|Context Request| D
    D -->|Embeddings| E[Neural Engine]
    C -->|Final Response| A
    
    style A fill:#2a3f5f,color:#fff
    style B fill:#5f2a3f,color:#fff
    style C fill:#3f5f2a,color:#fff
    style D fill:#2a5f3f,color:#fff
    style E fill:#5f3f2a,color:#fff
```

## 🔄 Setup Process Flow

```mermaid
sequenceDiagram
    participant U as User
    participant S as System
    participant DB as Database
    participant LLM as GroqAI
    
    U->>S: Initialize Application
    S->>DB: Check Existing Embeddings
    alt Embeddings Exist
        DB-->>S: Load Existing Data
    else No Embeddings
        S->>S: Generate New Embeddings
        S->>DB: Store Embeddings
    end
    S->>LLM: Test Connection
    LLM-->>S: Confirm Ready
    S->>U: Display Interface
    
    Note over U,S: System Ready for Queries
```



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- MicroLabs USA for pharmaceutical data
- Groq team for AI infrastructure
- LangChain community for framework support

---
<p align="center">
Made with 💊 by Siddharth Prakash
</p>
