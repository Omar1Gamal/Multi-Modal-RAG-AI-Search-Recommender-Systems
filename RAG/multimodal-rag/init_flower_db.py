#!/usr/bin/env python
"""
Initialize the flower image database for the multimodal RAG system.
This script adds all images from the ./images/ folder to ChromaDB.
"""

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import os
from pathlib import Path


class DummyEmbeddingFunction(EmbeddingFunction):
    """Lightweight embedding function that doesn't require downloads"""
    def __init__(self):
        pass
    
    def __call__(self, input: Documents) -> Embeddings:
        return [[0.1] * 384 for _ in input]


def initialize_flower_database():
    """Initialize the flower database with images from ./images/ folder"""
    
    print("=" * 60)
    print("Initializing Flower Image Database")
    print("=" * 60)
    
    # Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path="./data/flower.db")
    image_loader = ImageLoader()
    
    # Get or create the collection
    collection = chroma_client.get_or_create_collection(
        "flowers_collection",
        embedding_function=DummyEmbeddingFunction(),
        data_loader=image_loader,
    )
    
    # Find all images in the ./images/ folder
    images_path = Path("./images")
    if not images_path.exists():
        print(f"❌ Error: {images_path} folder not found!")
        return False
    
    image_files = sorted([f for f in images_path.glob("*.jpg")] + [f for f in images_path.glob("*.png")])
    
    if not image_files:
        print(f"❌ No images found in {images_path}")
        return False
    
    print(f"\nFound {len(image_files)} images:")
    for img in image_files:
        print(f"  - {img.name}")
    
    # Prepare data for insertion
    ids = [str(i) for i in range(len(image_files))]
    uris = [str(img) for img in image_files]
    metadatas = [
        {"source_file": img.name, "category": "flower"}
        for img in image_files
    ]
    
    # Add images to the collection
    print(f"\nAdding {len(image_files)} images to ChromaDB...")
    try:
        collection.add(ids=ids, uris=uris, metadatas=metadatas)
        print(f"✓ Successfully added {len(image_files)} images")
        print(f"  Database path: ./data/flower.db")
        print(f"  Collection: flowers_collection")
        print(f"  Total images in collection: {collection.count()}")
    except Exception as e:
        print(f"❌ Error adding images: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Database initialized successfully!")
    print("=" * 60)
    print("\nYou can now run the web UI:")
    print("  .venv/bin/streamlit run multimodal_rag_final_ui.py")
    print("\n" + "=" * 60 + "\n")
    
    return True


if __name__ == "__main__":
    import sys
    success = initialize_flower_database()
    sys.exit(0 if success else 1)
