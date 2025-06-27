# AST-Based Code Similarity Detection

A lightweight system for detecting code similarity using AST (Abstract Syntax Tree) graph representation combined with CodeBERT embeddings. This project provides a simple and efficient way to compare code files using both structural and semantic features with local JSON storage.

## 🎯 Project Overview

This system embeds code files using AST graph features combined with CodeBERT embeddings and stores the embeddings locally using JSON files. It's designed to help educators and code reviewers detect potential plagiarism by comparing code similarity using both structural (AST) and semantic (BERT) features.

## 🚀 Features

- **AST Graph Processing**: Convert Python code to graph representation using AST
- **Hybrid Embeddings**: Combine structural (AST) and semantic (CodeBERT) features
- **Local Storage**: JSON-based storage for simplicity and portability
- **Similarity Detection**: Multiple similarity metrics (cosine, euclidean, manhattan)
- **Easy Setup**: No external database required
- **Command Line Interface**: Simple CLI for common operations

## 📋 Architecture

### AST Graph Features (27 dimensions)
- **Structural Metrics**: Function count, class count, import count
- **Complexity Metrics**: Cyclomatic complexity, control flow count
- **Node Distribution**: Distribution of AST node types (functions, classes, loops, etc.)
- **Edge Distribution**: Distribution of graph edge types (contains, calls, uses, etc.)

### CodeBERT Features (768 dimensions)
- **Semantic Understanding**: Deep semantic representation of code
- **Context Awareness**: Understanding of code context and relationships
- **Multi-language Support**: Works with Python, Java, JavaScript, PHP, Ruby, Go

### Combined Embedding (795 dimensions)
- **Hybrid Approach**: Combines structural and semantic features
- **Better Similarity Detection**: More accurate similarity scores
- **Robust Performance**: Handles both syntactic and semantic similarities

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nagsujosh/code-smell/
   cd codeSmell
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Test AST Processing (No ML Dependencies)
```bash
python3 test_ast_only.py
```

### Run Full Demo
```bash
python3 main.py
```

### Command Line Interface

```bash
# Initialize the project
python -m src.cli init

# Embed a single file
python -m src.cli embed --file path/to/file.py --model codebert

# Compare two files
python -m src.cli compare --file1 path/to/file1.py --file2 path/to/file2.py

# Scan a directory for similar files
python -m src.cli scan --directory path/to/code/dir --model codebert

# List available models
python -m src.cli list-models

# Clean up old results
python -m src.cli cleanup --days 30
```

### Python API

```python
from src.models.embedder import CodeEmbedder
from src.similarity.calculator import SimilarityCalculator
from src.utils.json_storage import JSONStorage

# Initialize components with AST and BERT features
embedder = CodeEmbedder(
    model_name="codebert",
    use_ast_features=True,
    use_bert_features=True
)
calculator = SimilarityCalculator(threshold=0.8)
storage = JSONStorage()

# Embed a file (combines AST + BERT features)
embedding = embedder.embed_file("path/to/file.py")

# Get only AST features
ast_features = embedder.get_ast_features_only(code_text)

# Get only BERT features
bert_features = embedder.get_bert_features_only(code_text)

# Compare two embeddings
similarity = calculator.cosine_similarity(embedding1, embedding2)

# Save to storage
file_id = storage.save_code_file("path/to/file.py", content, "python")
embedding_id = storage.save_embedding(file_id, "codebert", embedding)
```

## 📁 Project Structure

```
codeSmell/
├── src/
│   ├── models/           # CodeBERT embedder
│   ├── similarity/       # Similarity calculation logic
│   └── utils/           # AST processor and JSON storage
├── data/                # Storage directory
├── main.py              # Main demo script
├── test_ast_only.py     # AST-only test (no ML)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 📊 Features

### Similarity Metrics

- **Cosine Similarity**: Most common for embeddings
- **Euclidean Similarity**: Distance-based comparison
- **Manhattan Similarity**: L1 distance comparison

### Storage Features

- **Local JSON Storage**: No external database required
- **Automatic Cleanup**: Remove old results and unused files
- **Statistics**: Track storage usage and file counts
- **Vector Storage**: Efficient numpy array storage for embeddings

## 🧪 Testing

The project includes basic functionality testing. You can test the system with:

```bash
# Test AST processing only (no ML dependencies)
python3 test_ast_only.py

# Test full embedding workflow
python3 main.py

# Test CLI commands
python -m src.cli embed --file src/cli.py --model codebert
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft Research for CodeBERT
- Hugging Face for the transformers library
- NetworkX for graph processing

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers. 