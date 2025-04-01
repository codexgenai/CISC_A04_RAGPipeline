# RAG Pipeline with Multi-Model Support

A flexible and efficient Retrieval-Augmented Generation (RAG) pipeline that supports multiple language models and provides enhanced document processing capabilities.

## Features

- **Multi-Model Support**:
  - Causal Language Models (OPT)
  - Sequence-to-Sequence Models (FLAN-T5)
  - Model-specific prompt templates and handling

- **Document Processing**:
  - PDF document ingestion
  - Text cleaning and preprocessing
  - Chunking and embedding generation
  - Vector database storage and retrieval

- **Performance Optimizations**:
  - CPU-optimized processing
  - Memory-efficient operations
  - Batch processing support
  - Caching mechanisms

- **Flexible Configuration**:
  - Model-specific configurations
  - Customizable parameters
  - Environment-based settings
  - Comprehensive logging

## Project Structure

```
.
├── classes/
│   ├── __init__.py
│   ├── config_manager.py
│   ├── document_processor.py
│   ├── embedding_preparer.py
│   ├── llm_client.py
│   ├── rag_query_processor.py
│   └── chromadb_retriever.py
├── data/
│   ├── raw_input/
│   ├── cleaned_text/
│   ├── embeddings/
│   └── vectordb/
├── config_causal.json
├── config_seq2seq.json
├── main.py
└── requirements.txt
```

## System Architecture

### Overview
The system follows a modular architecture with clear separation of concerns and data flow:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Document Input │     │  Text Processing│     │  Embedding Gen. │
│  (PDF Files)    │ ──> │  & Chunking     │ ──> │  & Storage     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Query Input    │     │  Context        │     │  Response      │
│  (User Query)   │ ──> │  Retrieval      │ ──> │  Generation    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Component Interactions

1. **Document Processing Pipeline**:
   ```
   PDF Files → Document Processor → Cleaned Text → Embedding Preparer → Vector DB
   ```
   - Document Processor handles PDF ingestion and text cleaning
   - Embedding Preparer generates embeddings for document chunks
   - Vector DB stores embeddings for retrieval

2. **Query Processing Pipeline**:
   ```
   User Query → RAG Query Processor → Context Retrieval → LLM Client → Response
   ```
   - RAG Query Processor formats queries and manages context
   - ChromaDB Retriever finds relevant document chunks
   - LLM Client generates responses using retrieved context

### Data Flow

1. **Document Flow**:
   ```
   raw_input/ → cleaned_text/ → embeddings/ → vectordb/
   ```
   - Raw PDFs are processed and cleaned
   - Cleaned text is chunked and embedded
   - Embeddings are stored in vector database

2. **Query Flow**:
   ```
   Query → Vector Search → Context → LLM → Response
   ```
   - Query is embedded and matched against stored vectors
   - Relevant context is retrieved and formatted
   - LLM generates response using context

### Model Integration

1. **Causal Models (OPT)**:
   ```
   Input → Tokenization → Model → Response
   ```
   - Direct text generation
   - Context-aware processing
   - Prompt template handling

2. **Seq2Seq Models (FLAN-T5)**:
   ```
   Input → Task Prefix → Tokenization → Model → Response
   ```
   - Task-specific processing
   - Structured input/output
   - Specialized prompt handling

### Configuration Management

```
┌─────────────────┐
│  Config Files   │
│  (causal/seq2seq)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Config Manager │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Component      │
│  Initialization │
└─────────────────┘
```

### Error Handling & Logging

```
┌─────────────────┐
│  Component      │
│  Operations     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Error Handler  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Logging System │
└─────────────────┘
```

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- ChromaDB
- PDF processing libraries
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses two main configuration files:

1. **config_causal.json**: For OPT model settings
2. **config_seq2seq.json**: For FLAN-T5 model settings

Key configuration parameters:
- Model-specific settings
- Directory paths
- Performance parameters
- Logging settings

## Usage

### Pipeline Steps

1. **Document Ingestion**:
```bash
python main.py step01_ingest --input_filename <filename> --model_type <causal|seq2seq>
```

2. **Embedding Generation**:
```bash
python main.py step02_generate_embeddings --input_filename <filename> --model_type <causal|seq2seq>
```

3. **Vector Storage**:
```bash
python main.py step03_store_vectors --input_filename <filename> --model_type <causal|seq2seq>
```

4. **Document Retrieval**:
```bash
python main.py step04_retrieve_chunks --query_args "<query>" --model_type <causal|seq2seq>
```

5. **Response Generation**:
```bash
python main.py step05_generate_response --query_args "<query>" --use_rag --model_type <causal|seq2seq>
```

### Example Usage

```bash
# Process a single document
python main.py step01_ingest --input_filename document.pdf --model_type causal

# Generate embeddings
python main.py step02_generate_embeddings --input_filename document.pdf --model_type causal

# Store vectors
python main.py step03_store_vectors --input_filename document.pdf --model_type causal

# Query the system
python main.py step05_generate_response --query_args "What is the main topic?" --use_rag --model_type causal
```

## Components

### 1. Document Processor
- Handles PDF document ingestion
- Performs text cleaning and preprocessing
- Manages document chunking

### 2. Embedding Preparer
- Generates embeddings for document chunks
- Supports multiple embedding models
- Optimizes memory usage

### 3. Vector Database Manager
- Stores and retrieves document embeddings
- Implements similarity search
- Manages context extraction

### 4. LLM Client
- Handles model initialization and inference
- Supports multiple model types
- Manages prompt templates

### 5. RAG Query Processor
- Processes user queries
- Retrieves relevant context
- Generates responses

## Performance Considerations

- **Memory Usage**:
  - Optimized for CPU processing
  - Efficient tensor handling
  - Batch processing support

- **Processing Speed**:
  - Parallel document processing
  - Caching mechanisms
  - Optimized vector search

## Error Handling

- Comprehensive error logging
- Graceful failure recovery
- Clear error messages
- Validation checks

## Testing

The project includes:
- Unit tests for each component
- Integration tests
- Performance benchmarks
- Error recovery tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify License]

## Acknowledgments

- Transformers library
- ChromaDB
- PyTorch
- Other dependencies and inspirations 