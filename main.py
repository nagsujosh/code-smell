#!/usr/bin/env python3
"""
Main script for AST-based code similarity detection.
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.embedder import CodeEmbedder
from src.similarity.calculator import SimilarityCalculator
from src.utils.json_storage import JSONStorage


def main():
    """Main function demonstrating the AST-based code embedding workflow."""
    print("AST-Based Code Similarity Detection")
    print("=" * 50)
    
    # Sample Python code files for testing
    code1 = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
    
    code2 = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""
    
    code3 = """
def calculate_area(radius):
    pi = 3.14159
    area = pi * radius * radius
    return area

def calculate_circumference(radius):
    pi = 3.14159
    circumference = 2 * pi * radius
    return circumference
"""
    
    try:
        # Initialize components
        print("1. Initializing embedder...")
        embedder = CodeEmbedder(
            model_name="codebert",
            use_ast_features=True,
            use_bert_features=True
        )
        
        print(f"   Model: {embedder.get_model_info()['name']}")
        print(f"   AST features: {embedder.get_model_info()['use_ast_features']}")
        print(f"   BERT features: {embedder.get_model_info()['use_bert_features']}")
        print(f"   Total embedding dimension: {embedder.get_embedding_dimension()}")
        
        # Initialize storage and calculator
        storage = JSONStorage()
        calculator = SimilarityCalculator(threshold=0.8)
        
        # Process each code sample
        code_samples = [
            ("bubble_sort.py", code1),
            ("quick_sort.py", code2),
            ("circle_calculator.py", code3)
        ]
        
        embeddings = []
        file_ids = []
        
        print("\n2. Processing code samples...")
        for filename, code in code_samples:
            print(f"   Processing {filename}...")
            
            # Save file to storage
            file_id = storage.save_code_file(filename, code, "python")
            file_ids.append(file_id)
            
            # Create embedding (AST + BERT)
            embedding = embedder.embed_text(code)
            embeddings.append(embedding)
            
            # Save embedding to storage
            embedding_id = storage.save_embedding(file_id, "codebert", embedding)
            
            print(f"     File ID: {file_id}")
            print(f"     Embedding ID: {embedding_id}")
            print(f"     Embedding shape: {embedding.shape}")
        
        # Calculate similarities
        print("\n3. Calculating similarities...")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                cosine_sim = calculator.cosine_similarity(embeddings[i], embeddings[j])
                euclidean_sim = calculator.euclidean_similarity(embeddings[i], embeddings[j])
                
                print(f"   {code_samples[i][0]} vs {code_samples[j][0]}:")
                print(f"     Cosine similarity: {cosine_sim:.4f}")
                print(f"     Euclidean similarity: {euclidean_sim:.4f}")
                print(f"     Are similar (cosine): {calculator.is_similar(embeddings[i], embeddings[j], 'cosine')}")
        
        # Show storage statistics
        print("\n4. Storage statistics:")
        stats = storage.get_stats()
        print(f"   Total files: {stats['total_files']}")
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   Storage size: {stats['storage_size_mb']:.2f} MB")
        
        print("\n✅ Workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 