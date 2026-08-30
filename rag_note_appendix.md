
---

## 三、Embedding（双路向量化）

### 架构定位

Ingestion Pipeline 的 **Stage 5**（Encoding），由 `BatchProcessor` 统一编排 Dense 和 Sparse 两条路径。

```
                    ┌──────────────────┐
                    │   BatchProcessor │
                    │  process(chunks) │
                    └────────┬─────────┘
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
    ┌──────────────────┐          ┌──────────────────────┐
    │   DenseEncoder   │          │    SparseEncoder     │
    │  embedding.embed │          │  jieba + BM25 stats  │
    │  (Batch API)     │          │  (本地计算，无 API)   │
    ├──────────────────┤          ├──────────────────────┤
    │ → 浮点向量       │          │ → term_frequencies   │
    │   [0.1, 0.2, ...]│          │ → doc_length         │
    └────────┬─────────┘          └──────────┬───────────┘
             │                               │
             └───────────┬───────────────────┘
                         ▼
              ┌──────────────────────┐
              │    BatchResult       │
              │  dense_vectors  ✅   │
              │  sparse_stats   ✅   │
              └──────────────────────┘
```

### DenseEncoder

**文件**：`src/ingestion/embedding/dense_encoder.py`

- 接收 `BaseEmbedding` 实例（依赖注入）
- 按 `batch_size` 分片调用 `embedding.embed()`
- 验证输出维度一致性

**`BaseEmbedding` 抽象**（`src/libs/embedding/base_embedding.py`）：
```python
class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: List[str], trace=None, **kwargs) -> List[List[float]]:
        """批量向量化。输入文本列表，输出等长向量列表。"""
```

**三个实现：**
- OpenAIEmbedding（远程 API，真批处理）
- AzureEmbedding（远程 API，真批处理）
- OllamaEmbedding（本地 Ollama，逐条调用）

### SparseEncoder

**文件**：`src/ingestion/embedding/sparse_encoder.py`

纯本地计算，无需外部 API：jieba 分词 + Counter 词频统计 → BM25 term_stats。

### BatchProcessor

编排 Dense + Sparse，产出 `BatchResult(dense_vectors, sparse_stats, ...)`。

### EmbeddingFactory

**文件**：`src/libs/embedding/embedding_factory.py`

工厂模式 + 模块加载时自动注册 provider。

### 增量嵌入

**当前状态：❌ 未实现**。目前只做文件级 SHA256 去重，无 chunk 级别增量复用。

---

## 四、Upsert & Storage（索引存储）

### 三路并行存储

Stage 6 同时写入三个后端：

```
6a: VectorUpserter  → ChromaDB      （Dense Vector + 正文 + Metadata）
6b: BM25Indexer     → JSON 文件     （倒排索引）
6c: ImageStorage    → SQLite + Disk （图片文件路径索引）
```

### 6a: VectorUpserter → ChromaDB

**文件**：`src/ingestion/storage/vector_upserter.py`

All-in-One 存储策略：每条记录包含 ID、Vector、完整文本、富 Metadata。ChunkID = `{path_hash[:8]}_{chunk_index:04d}_{text_hash[:8]}`（幂等、可追溯）。

**ChromaStore**（`src/libs/vector_store/chroma_store.py`）：通过 `collection.upsert(ids, embeddings, metadatas, documents)` 写入，查询时直接返回 text + metadata，无需二次查库。

### 6b: BM25Indexer

**文件**：`src/ingestion/storage/bm25_indexer.py`

标准 BM25 算法（k1=1.5, b=0.75），JSON 持久化，支持增量 add_documents 和 remove_document。

### 6c: ImageStorage

**文件**：`src/ingestion/storage/image_storage.py`

图片由 PdfLoader 提前保存，Stage 6c 只做 SQLite 索引注册。

### ID 同步关键

VectorUpserter 生成稳定 ChromaDB ID 后，回写给 sparse_stats，确保 BM25 查到的 ID 能在 ChromaDB 中匹配。

---

## 五、All-in-One 存储策略核查

| 要求 | 状态 |
|------|------|
| Dense Vector 持久化 | ✅ ChromaDB |
| Sparse Vector (BM25) | ✅ JSON 倒排索引 |
| Payload: 完整文本 | ✅ metadata.text + documents 双副本 |
| Payload: 富 Metadata | ✅ title/summary/tags... |
| 无需二次查库 | ✅ ChromaDB query 直接返回 |
| 原子幂等 | ✅ SHA256 内容 ID + INSERT OR REPLACE |

## 六、待扩展点

- LoaderFactory 按后缀自动派发（已实现，支持 PDF、Markdown；新增格式只需实现 `BaseLoader` 并 `LoaderFactory.register`）
- Chunk 级别内容哈希增量嵌入
- 更多 VectorStore 后端（Qdrant/Pinecone 接口已定义）
