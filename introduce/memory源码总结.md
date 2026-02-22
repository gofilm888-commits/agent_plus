# Memory 模块源码总结

> HelloAgents 分层记忆系统完整解读，涵盖基础数据结构、四种记忆类型、MemoryManager、嵌入模型、存储层与 RAG 流水线。

---

## 一、模块概览

### 1.1 定位与职责

`memory` 模块为 Agent 提供**分层记忆能力**，参考第 8 章架构设计，实现：

- **工作记忆**：短期、容量受限、会话级
- **情景记忆**：具体交互事件、时间序列、向量检索
- **语义记忆**：概念与知识、向量+知识图谱混合检索
- **感知记忆**：多模态（文本、图像、音频）、长期存储

### 1.2 目录结构

```
memory/
├── __init__.py          # 导出 MemoryManager、各记忆类型、存储
├── base.py              # MemoryItem、MemoryConfig、BaseMemory
├── manager.py           # MemoryManager 统一管理
├── embedding.py         # 嵌入模型（DashScope、Local、TF-IDF）
├── types/               # 记忆类型实现
│   ├── working.py      # WorkingMemory
│   ├── episodic.py     # EpisodicMemory
│   ├── semantic.py     # SemanticMemory
│   └── perceptual.py   # PerceptualMemory
├── storage/            # 存储层
│   ├── document_store.py  # SQLiteDocumentStore
│   ├── qdrant_store.py    # Qdrant 向量存储
│   └── neo4j_store.py     # Neo4j 图存储
└── rag/                 # RAG 流水线
    ├── pipeline.py     # create_rag_pipeline
    └── document.py     # 文档处理
```

### 1.3 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Integration Layer（MemoryTool、ContextBuilder 等）          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  MemoryManager（统一接口：add、retrieve、consolidate、forget）│
└─────────────────────────────────────────────────────────────┘
                              │
┌──────────────┬──────────────┬──────────────┬────────────────┐
│ WorkingMemory│ EpisodicMem  │ SemanticMem │ PerceptualMem  │
│ (内存+堆)    │ (SQLite+Qdrant)│ (Qdrant+Neo4j)│ (SQLite+Qdrant)│
└──────────────┴──────────────┴──────────────┴────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│  Storage Layer（SQLiteDocumentStore、Qdrant、Neo4j）         │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、基础层（base.py）

### 2.1 MemoryItem

```python
class MemoryItem(BaseModel):
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| id | str | 唯一标识 |
| content | str | 记忆内容 |
| memory_type | str | working/episodic/semantic/perceptual |
| user_id | str | 用户 ID |
| timestamp | datetime | 时间戳 |
| importance | float | 重要性 0–1 |
| metadata | Dict | 扩展元数据（session_id、modality 等） |

### 2.2 MemoryConfig

```python
class MemoryConfig(BaseModel):
    storage_path: str = "./memory_data"
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]
```

### 2.3 BaseMemory（抽象基类）

**抽象方法**：`add`、`retrieve`、`update`、`remove`、`has_memory`、`clear`、`get_stats`

**辅助方法**：
- `_generate_id()`：UUID
- `_calculate_importance(content, base_importance)`：基于长度、关键词计算重要性

---

## 三、记忆类型详解

### 3.1 WorkingMemory（工作记忆）

**特点**：纯内存、容量与 token 限制、TTL 过期、优先级堆

| 配置 | 默认值 | 说明 |
|------|--------|------|
| max_capacity | 10 | 最大条数 |
| max_tokens | 2000 | 最大 token |
| max_age_minutes | 120 | TTL（分钟） |

**存储**：`List[MemoryItem]` + `heapq` 优先级堆

**检索**：TF-IDF + 关键词匹配，混合分数 = 0.7×向量 + 0.3×关键词，再乘时间衰减与重要性权重

**遗忘**：`importance_based`、`time_based`、`capacity_based`，先执行 TTL 过期

**特有方法**：`get_recent`、`get_important`、`get_context_summary`、`get_all`

### 3.2 EpisodicMemory（情景记忆）

**特点**：具体交互事件、SQLite 权威存储 + Qdrant 向量索引、按 session 组织

**Episode 结构**：episode_id、user_id、session_id、timestamp、content、context、outcome、importance

**存储**：
- 内存：`episodes`、`sessions`（session_id → episode_ids）
- SQLite：`SQLiteDocumentStore` 权威
- Qdrant：向量索引

**检索**：向量检索 + 可选 time_range、importance_threshold 过滤，综合分 = 0.8×向量 + 0.2×新近性，再乘重要性权重

**特有方法**：`get_session_episodes`、`find_patterns`、`get_timeline`、`get_all`

### 3.3 SemanticMemory（语义记忆）

**特点**：概念与知识、Qdrant 向量 + Neo4j 知识图谱、spaCy 实体/关系提取

**Entity**：entity_id、name、entity_type、description、properties、frequency

**Relation**：from_entity、to_entity、relation_type、strength、evidence

**存储**：
- 内存：`semantic_memories`、`memory_embeddings`、`entities`、`relations`
- Qdrant：向量
- Neo4j：实体与关系

**检索**：向量检索 + 图检索，混合分 = 0.7×向量 + 0.3×图，再乘重要性权重；softmax 归一化得 probability

**NLP**：spaCy 多语言（zh_core_web_sm、en_core_web_sm），实体识别、词法分析写入 Neo4j

**特有方法**：`get_entity`、`search_entities`、`get_related_entities`、`export_knowledge_graph`

### 3.4 PerceptualMemory（感知记忆）

**特点**：多模态、SQLite + Qdrant（按模态分集合）、CLIP/CLAP 可选、哈希兜底

**Perception**：perception_id、data、modality、encoding、metadata

**模态**：text、image、audio、video、structured

**编码**：
- 文本：`get_text_embedder()`
- 图像：CLIP 或 `_image_encoder_hash`（确定性哈希向量）
- 音频：CLAP 或 `_audio_encoder_hash`

**存储**：SQLite 权威 + Qdrant 按模态分集合（`_perceptual_text`、`_perceptual_image`、`_perceptual_audio`）

**检索**：同模态向量检索，综合分 = 0.8×向量 + 0.2×新近性，再乘重要性权重

**特有方法**：`cross_modal_search`、`get_by_modality`、`generate_content`

---

## 四、MemoryManager

### 4.1 构造函数

```python
def __init__(
    self,
    config: Optional[MemoryConfig] = None,
    user_id: str = "default_user",
    enable_working: bool = True,
    enable_episodic: bool = True,
    enable_semantic: bool = True,
    enable_perceptual: bool = False
):
```

### 4.2 核心方法

| 方法 | 说明 |
|------|------|
| `add_memory(content, memory_type, importance, metadata, auto_classify)` | 自动分类、计算重要性、创建 MemoryItem、写入对应类型 |
| `retrieve_memories(query, memory_types, limit, min_importance, time_range)` | 从各类型检索，按重要性排序，截断 limit |
| `update_memory(memory_id, content, importance, metadata)` | 查找所在类型并更新 |
| `remove_memory(memory_id)` | 查找并删除 |
| `forget_memories(strategy, threshold, max_age_days)` | 调用各类型的 forget |
| `consolidate_memories(from_type, to_type, importance_threshold)` | 将高重要性短期记忆迁移到长期 |
| `get_memory_stats()` | 汇总各类型统计 |
| `clear_all_memories()` | 清空所有类型 |

### 4.3 自动分类

- `metadata.type` 优先
- 情景关键词：昨天、今天、上次、记得、发生、经历 → episodic
- 语义关键词：定义、概念、规则、知识、原理、方法 → semantic
- 否则 → working

### 4.4 重要性计算

- 基础 0.5
- 长度 > 100：+0.1
- 关键词（重要、关键、必须等）：+0.2
- metadata.priority：high +0.3，low -0.2
- 钳制到 [0, 1]

---

## 五、嵌入模块（embedding.py）

### 5.1 实现类

| 类 | 说明 |
|----|------|
| LocalTransformerEmbedding | sentence-transformers 或 transformers+torch |
| TFIDFEmbedding | sklearn TfidfVectorizer，需先 fit |
| DashScopeEmbedding | 通义千问 / OpenAI 兼容 REST |

### 5.2 环境变量

| 变量 | 说明 |
|------|------|
| EMBED_MODEL_TYPE | dashscope / local / tfidf |
| EMBED_MODEL_NAME | 模型名 |
| EMBED_API_KEY | API Key |
| EMBED_BASE_URL | REST 模式 base_url |

### 5.3 工厂与单例

- `create_embedding_model(model_type, **kwargs)`：创建指定类型
- `create_embedding_model_with_fallback(preferred_type)`：dashscope → local → tfidf 回退
- `get_text_embedder()`：线程安全单例
- `get_dimension(default=384)`：获取维度
- `refresh_embedder()`：强制重建

---

## 六、存储层

### 6.1 SQLiteDocumentStore

- **单例**：按 db_path 单例
- **表**：users、memories、concepts、memory_concepts、concept_relationships
- **方法**：add_memory、get_memory、search_memories、update_memory、delete_memory、add_document、get_document、get_database_stats

### 6.2 QdrantVectorStore / QdrantConnectionManager

- 向量存储，支持 add_vectors、search_similar、delete_memories、clear_collection、health_check
- QdrantConnectionManager：连接复用，避免重复连接

### 6.3 Neo4jGraphStore

- 图存储，实体与关系
- 方法：add_entity、add_relationship、find_related_entities、search_entities_by_name、get_entity_relationships、clear_all、health_check、get_stats

---

## 七、RAG 流水线（rag/pipeline.py）

### 7.1 create_rag_pipeline

- 创建 RAG 管道：文档解析 → 分块 → 嵌入 → Qdrant 存储
- 支持 MarkItDown 多格式（PDF、Office、图片、音频等）
- 增强 PDF 后处理

### 7.2 与 Memory 的关系

- RAG 独立于 memory 模块，供 RAGTool 使用
- EpisodicMemory、SemanticMemory 等使用相同的 embedding 与 Qdrant

---

## 八、遗忘策略

| 策略 | 说明 |
|------|------|
| importance_based | 删除 importance < threshold 的记忆 |
| time_based | 删除超过 max_age_days 的记忆 |
| capacity_based | 超出 max_capacity 时删除最低优先级 |

**实现**：各记忆类型实现 `forget(strategy, threshold, max_age_days)`，MemoryManager 汇总调用。

---

## 九、使用示例

### MemoryManager

```python
from hello_agents.memory import MemoryManager, MemoryConfig

config = MemoryConfig(storage_path="./my_memory")
manager = MemoryManager(
    config=config,
    user_id="user1",
    enable_working=True,
    enable_episodic=True,
    enable_semantic=True,
)

# 添加
mid = manager.add_memory("用户偏好：喜欢深色主题", memory_type="working", importance=0.8)

# 检索
results = manager.retrieve_memories("用户偏好", limit=5, min_importance=0.3)

# 整合
manager.consolidate_memories(from_type="working", to_type="episodic", importance_threshold=0.7)

# 遗忘
manager.forget_memories(strategy="importance_based", threshold=0.1)
```

### MemoryTool 集成

```python
from hello_agents.tools import MemoryTool
from hello_agents import FunctionCallAgent, HelloAgentsLLM, ToolRegistry

llm = HelloAgentsLLM(model="gpt-4")
registry = ToolRegistry()
registry.register_tool(MemoryTool(user_id="user1"))
agent = FunctionCallAgent(name="助手", llm=llm, tool_registry=registry)
agent.run("记住：我明天下午 3 点有会议")
```

---

## 十、环境变量与依赖

| 用途 | 环境变量 |
|------|----------|
| Qdrant | QDRANT_URL、QDRANT_API_KEY、QDRANT_COLLECTION、QDRANT_DISTANCE |
| Neo4j | NEO4J_URI、NEO4J_USERNAME、NEO4J_PASSWORD、NEO4J_DATABASE |
| Embedding | EMBED_MODEL_TYPE、EMBED_MODEL_NAME、EMBED_API_KEY、EMBED_BASE_URL |
| CLIP/CLAP | CLIP_MODEL、CLAP_MODEL |

**依赖**：sentence-transformers / transformers、scikit-learn、dashscope、qdrant-client、neo4j、spacy（可选）

---

## 十一、注意事项

1. **EpisodicMemory.get_all**：Episode 类仅有 `context` 属性，无 `metadata`，`metadata=episode.metadata` 会触发 AttributeError，应改为从 context/session_id/outcome 构建 metadata
2. **SemanticMemory**：依赖 Neo4j、Qdrant，缺一可能影响图检索或向量检索
3. **PerceptualMemory**：CLIP/CLAP 可选，缺依赖时退化为哈希编码，语义检索能力有限
4. **WorkingMemory**：纯内存，进程重启后丢失
5. **consolidate_memories**：需目标类型实现 `get_all`，部分类型可能未实现

---

*文档基于 hello_agents memory 模块源码整理。*
