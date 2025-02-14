"""
This module indexes documents from JSON files into a ChromaDB database using a local
embedding model. It processes documents from finance, insurance, and FAQ categories,
generates embeddings for each document, and stores them with associated metadata in
the ChromaDB collection.

Dependencies:
    - pysqlite3
    - sys
    - uuid
    - json
    - torch
    - chromadb
    - tqdm
    - transformers
"""

__import__("pysqlite3")
import sys
import uuid
import json
import torch
import chromadb
from tqdm import tqdm
from transformers import AutoModel

# 替換標準的 sqlite3 模組為 pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# 使用本地 embeddings 模型目錄
embedding_model_name = "../local_models/jina-embeddings-v3"

# 加載嵌入模型，並將其移至 GPU（如果可用）
embedding_model = AutoModel.from_pretrained(
    embedding_model_name, trust_remote_code=True
).to("cuda" if torch.cuda.is_available() else "cpu")


def indexing(index_name):
    """
    Indexes documents from a specified JSON file into the ChromaDB database.

    This function performs the following steps:
        1. Loads documents from the specified JSON file located in ../documents/{index_name}.json.
        2. Extracts texts and metadata from the loaded documents.
        3. Generates embeddings for the texts in batches of 200 using the pre-loaded embedding model.
        4. Connects to a persistent ChromaDB client and retrieves or creates a collection named "quickstart".
        5. Stores each document's text, metadata, and embedding into the ChromaDB collection with a unique UUID.

    Args:
        index_name (str): The name of the index/category to process (e.g., "finance", "insurance", "faq").

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        json.JSONDecodeError: If there is an error decoding the JSON file.
        chromadb.errors.ChromaDBException: If there is an error during ChromaDB operations.
    """
    print(f"Processing {index_name}...")
    json_path = f"../documents/{index_name}.json"

    # 加載 JSON 文檔
    try:
        with open(json_path, "rb") as f:
            docs = json.load(f)
        print(f"Number of documents in {index_name}: {len(docs)}")
    except FileNotFoundError:
        print(f"Error: The file {json_path} does not exist.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_path}: {e}")
        return

    # 提取文本和元數據
    texts = [doc.get("text", "") for doc in docs]
    metadatas = [doc.get("metadata", {}) for doc in docs]
    embeddings = []

    # 生成嵌入（批量處理，每200條文本一批）
    for i in range(0, len(texts), 200):
        batch_texts = texts[i : i + 200]
        batch_embeddings = embedding_model.encode(batch_texts, task="retrieval.passage")
        embeddings.extend(batch_embeddings)
        print(f"Processed embeddings for documents {i} to {i + len(batch_texts)}")

    print(f"Total embeddings generated: {len(embeddings)}")

    # 存儲到 Chroma 資料庫
    try:
        db = chromadb.PersistentClient(path="../ai_cup_2024")
        chroma_collection = db.get_or_create_collection("quickstart")

        print("Storing embeddings in ChromaDB...")
        for text, metadata, embedding in tqdm(
            zip(texts, metadatas, embeddings),
            total=len(texts),
            desc="Storing embeddings",
        ):
            chroma_collection.add(
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],  # 應該是列表
                ids=[str(uuid.uuid4())],
            )
        print(f"{index_name} index created successfully.")
    except chromadb.errors.ChromaDBException as e:
        print(f"Error with ChromaDB operations: {e}")


if __name__ == "__main__":
    """
    Main execution block.

    This block calls the indexing function for each of the following indices:
        - finance
        - insurance
        - faq

    It ensures that each category of documents is processed and indexed into the ChromaDB database.
    """
    indexing("finance")
    indexing("insurance")
    indexing("faq")