# Semantic Codebase Graph Engine (SCGE)

A language-agnostic system for analyzing software repositories, constructing semantic graph representations, and computing meaningful similarity scores between codebases.

## Overview

SCGE treats a codebase as a story told through structure and semantics. It produces machine-readable semantic graphs that can support visualization, search, and advanced analysis.

### Key Features

- **Repository Ingestion**: Supports local paths and GitHub URLs
- **Tree-sitter Parsing**: Accurate AST-based analysis for 10+ languages
- **Semantic Graphs**: Directed, labeled graphs representing code structure
- **Vector Embeddings**: Pre-trained CodeBERT embeddings for semantic similarity
- **Hybrid Similarity**: Combines structural and semantic analysis
- **Explainable Results**: Transparent scoring with detailed breakdowns

## System Architecture

### Pipeline Stages

```
Repository Ingestion
        |
        v
Language Detection
        |
        v
Tree-sitter Static Analysis  <-- Accurate AST parsing
        |
        v
Semantic Graph Construction
        |
        v
Semantic Encoding (CodeBERT)
        |
        v
Graph Storage
        |
        v
Similarity Analysis
        |
        v
Result Aggregation & Reporting
```

### Parsing Architecture

The system uses tree-sitter for accurate AST-based parsing across all supported languages:

| Language | Parser | Description |
|----------|--------|-------------|
| Python | Native AST | Python's built-in ast module |
| JavaScript | Tree-sitter | Classes, functions, imports, arrow functions |
| TypeScript | Tree-sitter | Full TS support including interfaces and types |
| Java | Tree-sitter | Classes, interfaces, enums, methods |
| Go | Tree-sitter | Structs, interfaces, functions, methods |
| Rust | Tree-sitter | Structs, enums, traits, impl blocks |
| C/C++ | Tree-sitter | Classes, structs, functions, namespaces |
| Ruby | Tree-sitter | Classes, modules, methods |
| PHP | Tree-sitter | Classes, interfaces, traits, functions |
| Kotlin | Regex | Fallback regex-based (tree-sitter optional) |

### Graph Data Model

#### Node Types

| Type | Description |
|------|-------------|
| `repository` | Root node representing the entire codebase |
| `file` | Source file in the repository |
| `module` | Module or package |
| `class` | Class, struct, interface, or trait definition |
| `function` | Function or method |
| `external_dependency` | External library or package |

#### Edge Types

| Type | Description |
|------|-------------|
| `contains` | Parent-child containment relationship |
| `defines` | Entity definition relationship |
| `calls` | Function call relationship |
| `imports` | Import/dependency relationship |
| `depends_on` | General dependency |
| `inherits` | Class inheritance |
| `implements` | Interface implementation |

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- Git (for remote repository cloning)

### Install from Source

```bash
git clone <repository-url>
cd codeSmell
pip install -e .
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

This installs tree-sitter and all language grammars automatically.

## Usage

### Command Line Interface

#### Compare Two Repositories

```bash
# Compare local repositories
python main.py compare ./repo1 ./repo2

# Compare remote repositories
python main.py compare https://github.com/user/repo1 https://github.com/user/repo2

# Save report to specific path
python main.py compare ./repo1 ./repo2 -o data/reports/my_comparison

# Adjust similarity weights
python main.py compare ./repo1 ./repo2 --structural-weight 0.3 --semantic-weight 0.7
```

#### Analyze Single Repository

```bash
python main.py analyze ./my-project
python main.py analyze https://github.com/user/repo -o data/reports/analysis
```

#### Other Commands

```bash
# Initialize configuration file
python main.py init

# List supported languages
python main.py list-languages

# Show storage statistics
python main.py storage-stats

# Clean up old graphs
python main.py cleanup --days 30
```

### Python API

```python
from src.engine import SemanticCodebaseEngine, compare_repositories

# Using the engine
engine = SemanticCodebaseEngine()
result = engine.compare("./repo1", "./repo2")

print(f"Overall Similarity: {result['similarity']['overall']:.2%}")
print(f"Structural: {result['similarity']['structural']:.2%}")
print(f"Semantic: {result['similarity']['semantic']:.2%}")

# Convenience function
result = compare_repositories("./repo1", "https://github.com/user/repo2")
```

## Example Comparison Results

Real-world comparison results from analyzing popular open-source libraries using tree-sitter based parsing:

| Comparison | Overall | Structural | Semantic | Category |
|------------|---------|------------|----------|----------|
| **FastAPI vs Flask** | **60.11%** | 55.88% | 62.93% | Web Frameworks (Python) |
| **winston vs log4js** | **57.22%** | 69.54% | 49.01% | Logging (JavaScript) |
| **axios vs node-fetch** | **42.64%** | 59.94% | 31.11% | HTTP Clients (JavaScript) |
| **lodash vs underscore** | **30.69%** | 57.93% | 12.53% | Utility Libraries (JavaScript) |

### Key Insights

- **FastAPI vs Flask (60.11%)**: Higher similarity with tree-sitter parsing detecting more shared patterns in Python web frameworks. Both use decorators, routing, and request handling.

- **winston vs log4js (57.22%)**: Moderate similarity between logging libraries. Tree-sitter accurately extracts the class hierarchies and configuration patterns.

- **axios vs node-fetch (42.64%)**: Lower semantic similarity (31.11%) indicates different implementation approaches despite similar structural organization for HTTP operations.

- **lodash vs underscore (30.69%)**: Low semantic similarity (12.53%) reflects that while both are utility libraries, they have evolved with significantly different implementations.

Example reports are included in `data/reports/` for reference.

## Similarity Methodology

### Structural Similarity

Computed from:
- **Node Type Distribution**: Cosine similarity of node type counts
- **Edge Type Distribution**: Cosine similarity of edge type counts
- **Dependency Overlap**: Jaccard index of external dependencies
- **Topology Metrics**: Graph density, clustering coefficient, depth

### Semantic Similarity

Computed from:
- **Function-Level**: Embedding similarity of functions/methods
- **Class-Level**: Embedding similarity of classes
- **File-Level**: Aggregated file embeddings

Uses CodeBERT embeddings and greedy optimal matching to pair similar entities.

### Hybrid Score

```
overall = (structural_weight * structural_score) + (semantic_weight * semantic_score)
```

Default weights: structural=0.4, semantic=0.6

## Configuration

Create a configuration file with `python main.py init`:

```json
{
  "ingestion": {
    "ignore_patterns": ["*.pyc", "__pycache__", ".git", "node_modules"],
    "max_file_size": 1048576,
    "clone_depth": 1
  },
  "embedding": {
    "model_name": "microsoft/codebert-base",
    "max_token_length": 512,
    "batch_size": 32
  },
  "similarity": {
    "structural_weight": 0.4,
    "semantic_weight": 0.6,
    "similarity_threshold": 0.7
  }
}
```

## Project Structure

```
src/
  core/           # Pipeline, configuration, exceptions
  ingestion/      # Repository ingestion (local/remote)
  analysis/       # Language detection and static analysis
    languages/    # Tree-sitter based language analyzers
  graph/          # Semantic graph construction
  embedding/      # Semantic payload extraction and encoding
  storage/        # Graph storage and persistence
  similarity/     # Structural, semantic, and hybrid analysis
  reporting/      # Report generation and formatting
  utils/          # Utilities (logging, validation)
  engine.py       # Main engine orchestrating the pipeline
  cli.py          # Command-line interface
```

## Adding Language Support

To add support for a new language using tree-sitter:

1. Install the tree-sitter grammar: `pip install tree-sitter-<language>`

2. Create a new analyzer in `src/analysis/languages/`:

```python
import tree_sitter_mylang as ts_mylang
from tree_sitter import Language, Parser

from src.analysis.registry import AnalyzerRegistry
from src.analysis.languages.base_treesitter_analyzer import BaseTreeSitterAnalyzer

@AnalyzerRegistry.register
class MyLanguageAnalyzer(BaseTreeSitterAnalyzer):
    LANGUAGE = "mylang"
    SUPPORTED_EXTENSIONS = [".ml"]

    def _initialize_parser(self) -> bool:
        try:
            self._language = Language(ts_mylang.language())
            self._parser = Parser(self._language)
            return True
        except Exception:
            return False

    def _extract_classes(self, tree, content_bytes):
        # Implement class extraction using tree-sitter queries
        pass

    def _extract_functions(self, tree, content_bytes):
        # Implement function extraction using tree-sitter queries
        pass

    def _extract_imports(self, tree, content_bytes):
        # Implement import extraction using tree-sitter queries
        pass
```

3. Import the analyzer in `src/analysis/languages/__init__.py`

4. The analyzer will be automatically discovered and used

## Supported Languages

| Language | Parser Type | Extensions |
|----------|-------------|------------|
| Python | AST (native) | `.py`, `.pyw`, `.pyi` |
| JavaScript | Tree-sitter | `.js`, `.jsx`, `.mjs`, `.cjs` |
| TypeScript | Tree-sitter | `.ts`, `.tsx`, `.mts`, `.cts` |
| Java | Tree-sitter | `.java` |
| Go | Tree-sitter | `.go` |
| Rust | Tree-sitter | `.rs` |
| C | Tree-sitter | `.c`, `.h` |
| C++ | Tree-sitter | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx` |
| Ruby | Tree-sitter | `.rb`, `.rake`, `.gemspec` |
| PHP | Tree-sitter | `.php`, `.phtml` |
| Kotlin | Regex (fallback) | `.kt`, `.kts` |

## Known Limitations

1. **Static Analysis Only**: Runtime behavior is not considered
2. **No Variable-Level Data Flow**: Fine-grained data dependencies not tracked
3. **Tree-sitter Coverage**: Some language-specific constructs may not be fully captured
4. **No Custom Model Training**: Uses pre-trained CodeBERT embeddings only
5. **Memory Constraints**: Large repositories may require significant RAM

## End-to-End Example

```bash
# Compare two web frameworks
python main.py compare https://github.com/tiangolo/fastapi https://github.com/pallets/flask \
    -o data/reports/fastapi_vs_flask

# View the results
cat data/reports/fastapi_vs_flask.txt
```

Expected output:
```
====================================================================================================
                                Repository Similarity Analysis Report
====================================================================================================

Report Generated: 2024-12-23 19:06:03 UTC
Pipeline ID: abc123

====================================================================================================
                                        EXECUTIVE SUMMARY
====================================================================================================

  OVERALL SIMILARITY SCORE: 0.6011 (60.11%)

  Interpretation: Moderate Similarity - Some common patterns

  Component Scores:
    Structural Similarity: 0.5588 (55.88%)
    Semantic Similarity:   0.6293 (62.93%)

  Overall   [############################################################..............................] 60.11%
  Structural[########################################################..................................] 55.88%
  Semantic  [###############################################################...........................] 62.93%
...
```
