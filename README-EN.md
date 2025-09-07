# Qwen3-8B 4-bit Inference and Small Context RAG System

English | [中文](./README.md)

This is a 4-bit quantized inference system based on Qwen/Qwen3-8B model, integrated with ChromaDB vector database and LlamaIndex framework for efficient Retrieval-Augmented Generation (RAG).

## Tech Stack

- **LLM Model**: Qwen/Qwen3-8B (4-bit quantized)
- **Embedding Model**: google/embeddinggemma-300m
- **Vector Database**: ChromaDB
- **RAG Framework**: LlamaIndex
- **Quantization**: bitsandbytes

## Features

- 🚀 4-bit quantized inference with low memory usage
- 📚 Efficient vector retrieval based on ChromaDB
- 🔍 LlamaIndex-powered RAG pipeline
- 🎯 Small context window optimization
- ⚡ Fast response and high throughput

## Project Structure

```
├── src/
│   ├── models/          # Model loading and inference
│   ├── embeddings/      # Embedding model services
│   ├── vectordb/        # ChromaDB vector database
│   ├── rag/             # RAG query processing
│   └── utils/           # Utility functions
├── config/              # Configuration files
├── tests/               # Test code
├── examples/            # Usage examples
└── data/                # Data files
```

## System Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- At least 8GB RAM
- At least 10GB disk space

## Installation

### Windows Users

1. Run the automated installation script:
```cmd
install.bat
```

2. Or install manually:
```cmd
# Create virtual environment
python -m venv venv
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Linux/Mac Users

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Interactive Usage

```bash
# Start interactive RAG system
python main.py --mode interactive

# Or use Windows script
run_example.bat
```

### 2. Programming Interface

```python
from src.rag.query_processor import RAGQueryProcessor

# Initialize RAG system
rag_processor = RAGQueryProcessor(
    llm_model_name="Qwen/Qwen3-8B",
    embedding_model_name="google/embeddinggemma-300m",
    use_llama_index=True
)

# Add documents
documents = [
    {"text": "Python is a programming language", "metadata": {"topic": "programming"}},
    {"text": "Machine learning is a branch of AI", "metadata": {"topic": "AI"}}
]
rag_processor.add_documents(documents)

# Query
result = rag_processor.query("What is Python?")
print(f"Answer: {result['response']}")
```

### 3. Command Line Usage

```bash
# Single query
python main.py --mode batch --query "What is artificial intelligence?"

# Add documents
python main.py --documents doc1.txt doc2.txt

# Show system information
python main.py --info

# Clear documents
python main.py --clear
```

## Examples

### Run Basic Example
```bash
python examples/basic_usage.py
```

### Run Document Processing Example
```bash
python examples/document_processing.py
```

## Configuration

System configuration file is located at `config/config.yaml`:

```yaml
# Model configuration
model:
  llm_model_name: "Qwen/Qwen3-8B"
  embedding_model_name: "google/embeddinggemma-300m"
  device: "auto"

# RAG configuration
rag:
  use_llama_index: true
  max_context_length: 1500
  top_k_retrieval: 5
  similarity_threshold: 0.7

# Generation configuration
generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
```

### Environment Variable Configuration

You can override configuration with environment variables:

```bash
set RAG_SYSTEM_LLM_MODEL=your-model-name
set RAG_SYSTEM_DEVICE=cuda
set RAG_SYSTEM_PERSIST_DIRECTORY=./custom_data
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python tests/test_rag_system.py
```

## Performance Optimization

1. **GPU Acceleration**: Set `device: "cuda"` to use GPU
2. **Memory Optimization**: Adjust `chunk_size` and `max_context_length`
3. **Batch Processing**: Use `batch_size` parameter to optimize embedding computation
4. **Caching**: Set `cache_dir` to cache model files

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `chunk_size` or use CPU mode
2. **Slow Model Download**: Set HuggingFace mirror or use local models
3. **CUDA Error**: Check CUDA version compatibility

### View Logs

Log files are located at `logs/rag_system.log` for detailed runtime information.

## License

MIT License