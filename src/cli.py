#!/usr/bin/env python3
"""
Command-line interface for code similarity detection.
"""

import click
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.embedder import CodeEmbedder
from src.similarity.calculator import SimilarityCalculator
from src.utils.json_storage import JSONStorage
from src.utils.file_utils import FileUtils
from src.utils.config import Config
from src.models.model_registry import ModelRegistry


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Code Similarity Detection CLI."""
    pass


@cli.command()
@click.option('--file', '-f', required=True, help='Path to the code file to embed')
@click.option('--model', '-m', default='codebert', help='Model to use for embedding')
@click.option('--output', '-o', help='Output file for embedding (optional)')
def embed(file, model, output):
    """Embed a code file using the specified model."""
    try:
        # Validate file exists
        file_path = Path(file)
        if not file_path.exists():
            click.echo(f"Error: File {file} does not exist", err=True)
            return
        
        # Initialize embedder and storage
        embedder = CodeEmbedder(
            model_name=model,
            use_ast_features=True,
            use_bert_features=True
        )
        storage = JSONStorage()
        
        click.echo(f"Using model: {embedder.get_model_info()['name']}")
        click.echo(f"AST features: {embedder.get_model_info()['use_ast_features']}")
        click.echo(f"BERT features: {embedder.get_model_info()['use_bert_features']}")
        click.echo(f"Total embedding dimension: {embedder.get_embedding_dimension()}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Detect language
        file_utils = FileUtils()
        language = file_utils.detect_language(file_path)
        
        # Save file to storage
        file_id = storage.save_code_file(str(file_path), content, language)
        
        # Embed file
        click.echo(f"Embedding {file_path.name}...")
        embedding = embedder.embed_file(file_path)
        
        # Save embedding to storage
        embedding_id = storage.save_embedding(file_id, model, embedding)
        
        click.echo(f"File ID: {file_id}")
        click.echo(f"Embedding ID: {embedding_id}")
        click.echo(f"Embedding shape: {embedding.shape}")
        click.echo(f"Embedding dimension: {embedding.shape[-1]}")
        
        # Save embedding if output specified
        if output:
            output_path = Path(output)
            saved_path = file_utils.save_embedding(embedding, file_path, model, output_path.parent)
            click.echo(f"Embedding saved to: {saved_path}")
        
        click.echo("Embedding completed successfully!")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@cli.command()
@click.option('--file1', '-f1', required=True, help='Path to first code file')
@click.option('--file2', '-f2', required=True, help='Path to second code file')
@click.option('--model', '-m', default='codebert', help='Model to use for comparison')
@click.option('--method', default='cosine', help='Similarity method (cosine, euclidean, manhattan)')
@click.option('--threshold', '-t', default=0.8, help='Similarity threshold')
def compare(file1, file2, model, method, threshold):
    """Compare two code files for similarity."""
    try:
        # Validate files exist
        file1_path = Path(file1)
        file2_path = Path(file2)
        
        if not file1_path.exists():
            click.echo(f"Error: File {file1} does not exist", err=True)
            return
        
        if not file2_path.exists():
            click.echo(f"Error: File {file2} does not exist", err=True)
            return
        
        # Initialize components
        embedder = CodeEmbedder(
            model_name=model,
            use_ast_features=True,
            use_bert_features=True
        )
        calculator = SimilarityCalculator(threshold=threshold)
        storage = JSONStorage()
        
        click.echo(f"Using model: {embedder.get_model_info()['name']}")
        click.echo(f"AST features: {embedder.get_model_info()['use_ast_features']}")
        click.echo(f"BERT features: {embedder.get_model_info()['use_bert_features']}")
        click.echo(f"Total embedding dimension: {embedder.get_embedding_dimension()}")
        click.echo(f"Similarity method: {method}")
        click.echo(f"Threshold: {threshold}")
        
        # Embed files
        click.echo(f"Embedding {file1_path.name}...")
        embedding1 = embedder.embed_file(file1_path)
        
        click.echo(f"Embedding {file2_path.name}...")
        embedding2 = embedder.embed_file(file2_path)
        
        # Calculate similarity
        if method == 'cosine':
            similarity = calculator.cosine_similarity(embedding1, embedding2)
        elif method == 'euclidean':
            similarity = calculator.euclidean_similarity(embedding1, embedding2)
        elif method == 'manhattan':
            similarity = calculator.manhattan_similarity(embedding1, embedding2)
        else:
            click.echo(f"Error: Unknown similarity method {method}", err=True)
            return
        
        # Determine if similar
        is_similar = calculator.is_similar(embedding1, embedding2, method)
        
        # Display results
        click.echo(f"\nSimilarity Results:")
        click.echo(f"  {file1_path.name} vs {file2_path.name}")
        click.echo(f"  Similarity score: {similarity:.4f}")
        click.echo(f"  Are similar: {is_similar}")
        click.echo(f"  Method: {method}")
        click.echo(f"  Threshold: {threshold}")
        
        # Calculate additional metrics
        additional_metrics = calculator.compare_embeddings(embedding1, embedding2)
        click.echo(f"\nAdditional metrics:")
        for method_name, score in additional_metrics.items():
            click.echo(f"  {method_name}: {score:.4f}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@cli.command()
@click.option('--directory', '-d', required=True, help='Directory containing code files')
@click.option('--model', '-m', default='codebert', help='Model to use')
@click.option('--threshold', '-t', default=0.8, help='Similarity threshold')
@click.option('--output', '-o', help='Output file for results')
def scan(directory, model, threshold, output):
    """Scan a directory for similar code files."""
    try:
        # Validate directory
        dir_path = Path(directory)
        if not dir_path.exists():
            click.echo(f"Error: Directory {directory} does not exist", err=True)
            return
        
        # Find code files
        file_utils = FileUtils()
        code_files = file_utils.find_code_files(dir_path)
        
        if not code_files:
            click.echo(f"No code files found in {directory}")
            return
        
        click.echo(f"Found {len(code_files)} code files")
        
        # Initialize components
        embedder = CodeEmbedder(
            model_name=model,
            use_ast_features=True,
            use_bert_features=True
        )
        calculator = SimilarityCalculator(threshold=threshold)
        storage = JSONStorage()
        
        click.echo(f"Using model: {embedder.get_model_info()['name']}")
        click.echo(f"AST features: {embedder.get_model_info()['use_ast_features']}")
        click.echo(f"BERT features: {embedder.get_model_info()['use_bert_features']}")
        click.echo(f"Total embedding dimension: {embedder.get_embedding_dimension()}")
        
        # Embed all files and save to storage
        file_ids = []
        embeddings = []
        
        for file_path in code_files:
            try:
                click.echo(f"Embedding {file_path.name}...")
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Detect language
                language = file_utils.detect_language(file_path)
                
                # Save file to storage
                file_id = storage.save_code_file(str(file_path), content, language)
                file_ids.append(file_id)
                
                # Embed file
                embedding = embedder.embed_file(file_path)
                embeddings.append(embedding)
                
                # Save embedding to storage
                storage.save_embedding(file_id, model, embedding)
                
            except Exception as e:
                click.echo(f"Error embedding {file_path.name}: {str(e)}")
                continue
        
        # Find similar pairs
        click.echo("Finding similar files...")
        similar_pairs = calculator.find_similar_pairs(embeddings, code_files, method='cosine')
        
        # Save results to storage
        for file_id in file_ids:
            similar_files = []
            for pair in similar_pairs:
                if pair['file1'] == file_id or pair['file2'] == file_id:
                    similar_files.append({
                        'file_id': pair['file2'] if pair['file1'] == file_id else pair['file1'],
                        'similarity_score': pair['similarity'],
                        'method': 'cosine'
                    })
            
            if similar_files:
                storage.save_similarity_result(file_id, model, similar_files)
        
        # Display results
        click.echo(f"\nFound {len(similar_pairs)} similar pairs:")
        for pair in similar_pairs:
            click.echo(f"  {pair['file1']} vs {pair['file2']}: {pair['similarity']:.4f}")
        
        # Save to output file if specified
        if output:
            import json
            results = {
                'similar_pairs': similar_pairs,
                'total_files': len(code_files),
                'model': model,
                'threshold': threshold
            }
            
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            
            click.echo(f"Results saved to: {output}")
        
        click.echo("Scan completed successfully!")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@cli.command()
def list_models():
    """List all available models."""
    try:
        models = ModelRegistry.list_models()
        
        click.echo("Available models:")
        click.echo("-" * 80)
        
        for name, config in models.items():
            click.echo(f"Name: {name}")
            click.echo(f"  Model: {config.name}")
            click.echo(f"  ID: {config.model_id}")
            click.echo(f"  Type: {config.type.value}")
            click.echo(f"  Languages: {', '.join(config.languages)}")
            click.echo(f"  Embedding dim: {config.embedding_dim}")
            click.echo(f"  Max length: {config.max_length}")
            click.echo(f"  Size: {config.size_mb}MB" if config.size_mb else "  Size: Unknown")
            click.echo(f"  Requires GPU: {config.requires_gpu}")
            click.echo(f"  Description: {config.description}")
            click.echo()
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@cli.command()
def init():
    """Initialize the project (create necessary directories and files)."""
    try:
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Initialize JSON storage
        storage = JSONStorage()
        
        # Get storage stats
        stats = storage.get_stats()
        
        click.echo("Project initialized successfully!")
        click.echo(f"Data directory: {data_dir.absolute()}")
        click.echo(f"Storage stats: {stats}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@cli.command()
@click.option('--days', default=30, help='Remove results older than this many days')
def cleanup(days):
    """Clean up old results and unused files."""
    try:
        storage = JSONStorage()
        storage.cleanup_old_results(days)
        
        stats = storage.get_stats()
        click.echo(f"Cleanup completed! Storage stats: {stats}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


if __name__ == '__main__':
    cli() 