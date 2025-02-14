"""
This script processes a set of questions by retrieving relevant context from a ChromaDB database,
reranking the retrieved documents using a reranker model, and generating answers using an Ollama
chat model. The final predictions are saved to a JSON file.

Dependencies:
    - pysqlite3
    - sys
    - uuid
    - json
    - torch
    - chromadb
    - tqdm
    - ollama
    - transformers
"""

__import__("pysqlite3")
import sys

# 替換標準的 sqlite3 模組為 pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import uuid
import json
import torch
import chromadb
from tqdm import tqdm
from ollama import chat
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification

# 使用本地 reranker 模型目錄
reranker_model_name = "../local_models/jina-reranker-v2-base-multilingual"
# reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    reranker_model_name, trust_remote_code=True
).to("cuda" if torch.cuda.is_available() else "cpu")

# 使用本地 embeddings 模型目錄
embedding_model_name = "../local_models/jina-embeddings-v3"
# embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(
    embedding_model_name, trust_remote_code=True
).to("cuda" if torch.cuda.is_available() else "cpu")

# 定義聊天模板
template = """
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query_str}
Answer:
"""

# 連接到 ChromaDB 並獲取集合
db = chromadb.PersistentClient(path="../ai_cup_2024")
chroma_collection = db.get_or_create_collection("quickstart")

# 加載問題
with open("../questions_example.json", "rb") as f:
    questions = json.load(f)["questions"]

final_pred = {"answers": []}

# 處理每個問題
for question in tqdm(questions):
    query = question["query"]
    sources = question["source"]
    category = question["category"]
    
    # 生成查詢嵌入
    embedding = embedding_model.encode(query, task="retrieval.query")
    
    # 從 ChromaDB 查詢相關文檔
    results = chroma_collection.query(
        query_embeddings=embedding,
        n_results=30,
        where={
            "$and": [
                {"category": category},
                {"source": {"$in": [str(s) for s in sources]}},
            ]
        },
    )
    
    # 使用 reranker 模型重新排序查詢結果
    final_results = reranker_model.rerank(
        query=query, documents=results["documents"][0], top_n=4
    )
    
    # 組合上下文
    contexts = [result["document"] for result in final_results]
    contexts = "\n\n".join(contexts)
    
    # 使用聊天模型生成回答
    response = chat(
        model="kenneth85/llama-3-taiwan:latest",
        options={"temperature": 0},
        messages=[
            {
                "role":"system",
                "content":"If you don't know the answer, please return '不知道'.",
            },
            {
                "role": "user",
                "content": template.format(context_str=contexts, query_str=query),
            }
        ],
    )
    
    # 添加回答到最終結果
    final_pred["answers"].append(
        {
            "qid": question["qid"],
            "Document1":contexts[0],
            "Document2":contexts[1],
            "Document3":contexts[2],
            "Document4":contexts[3],
            "generate": response["message"]["content"],
        }
    )

# 保存最終預測結果到 JSON 文件
with open("../final_pred.json", "w", encoding="utf8") as f:
    json.dump(final_pred, f, ensure_ascii=False, indent=4)
print("Done!")
