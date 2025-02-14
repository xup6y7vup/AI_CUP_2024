**專案結構**
```
./
├── Preprocess/
│   ├── build_json.py           # 建立 Documents JSON 文件
│   └── indexing.py             # 建立向量資料庫
├── Model/
│   └── llm_generate.py         # LLM 回答的程式 
├── local_models/
│   ├── jina-embeddings-v3      # Embeddings 模型
│   └── jina-reranker-v2-base-multilingual  # Rerank 模型
├── documents/
│   ├── faq.json
│   ├── finance.json
│   └── insurance.json
├── finance_markdown/            # Finance Markdown 資料集
├── insurance_markdown/         # Insurance Markdown 資料集
├── ai_cup_2024/                # 向量資料庫儲存的資料夾
├── requirements.txt
└── README.md
```
本專案使用 **Python 3.8.10**。請依照以下步驟進行：
使用之前請先安裝套件
```
pip install -r requirements.txt
```
### 模型資訊
本專案所使用的模型如下：
1. **Embeddings 模型**: jinaai/jina-embeddings-v3
1. **Rerank 模型**: jinaai/jina-reranker-v2-base-multilingual
1. **LLM**: kenneth85/llama-3-taiwan:latest
## 執行順序
請依照以下順序執行各程式，以完成整個流程：
1. build_json.py
1. indexing.py
1. llm_generate.py
