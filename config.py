# 配置文件（API Key等）
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.resolve()

# ==================== API 配置 ====================
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_API_BASE = os.getenv("ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")

# ==================== Redis 配置 ====================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ==================== 向量库配置 ====================
VECTOR_STORE_TYPE = "chroma"
CHROMA_QA_DIR = PROJECT_ROOT / "chroma_qa_db"
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "D:/FAISS/faiss_db")
PARENT_FAISS_DIR = os.getenv("PARENT_FAISS_DIR", "D:/FAISS/faiss_parent_db")
PARENT_DOCSTORE_DIR = PROJECT_ROOT / "parent_docstore"

# ==================== 模型路径 ====================
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "glm-4-air")   # 建议从 flash 升级到 air

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "D:/model/m3e-large",
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "D:/model/bge-reranker-large",
)

SAFETY_MODEL = os.getenv(
    "SAFETY_MODEL",
    "D:/model/Unified_Prompt_Guard",
)

SAFETY_MODEL_ENABLED = os.getenv("SAFETY_MODEL_ENABLED", "true").lower() == "true"

# ==================== 缓存 ====================
CACHE_ENABLED = True
LLM_CACHE_BACKEND = "memory"
EMBEDDING_CACHE_SIZE = 10000
ROUTER_USE_LLM = True

# ==================== 检索深度配置 ====================
RETRIEVAL_DEPTH_CONFIG = {
    "basic": {
        "top_k": 2,
        "use_hyde": False,
        "use_bm25": True,
        "use_reranker": False,
        "use_parent": False,
    },
    "standard": {
        "top_k": 4,
        "use_hyde": False,
        "use_bm25": True,
        "use_reranker": False,
        "use_parent": False,
    },
    "deep": {
        "top_k": 5,
        "use_hyde": True,
        "use_bm25": True,
        "use_reranker": True,
        "use_parent": True,
    },
}

USE_LLM_FOR_DEPTH = True
USE_PARENT_RETRIEVER = True

# ==================== 父文档切分配置 ====================
CHILD_CHUNK_SIZE = 300
CHILD_OVERLAP = 30
PARENT_CHUNK_SIZE = 1200
PARENT_OVERLAP = 100

# ==================== Token 预算 ====================
TOKEN_TRACK_ENABLED = True
TOKEN_BUDGET_PER_SESSION = 100_000
TOKEN_BUDGET_TOTAL = 1_000_000
TOKEN_WARN_RATIO = 0.8

# ==================== 医疗数据集分类配置 ====================
# 数据集来源：A Hospital Medical Wiki Dataset
# 每篇文章的 categories 字段是分类标签列表

# 是否启用基于分类的元数据过滤
ENABLE_CATEGORY_FILTER = os.getenv("ENABLE_CATEGORY_FILTER", "true").lower() == "true"

# 过滤检索时先多召回多少倍（先取 top_k * N，再按分类过滤到 top_k）
CATEGORY_FILTER_RECALL_MULTIPLIER = int(
    os.getenv("CATEGORY_FILTER_RECALL_MULTIPLIER", "3")
)

# 过滤模式下最少保留的文档数，不足时自动放宽过滤
CATEGORY_FILTER_MIN_RESULTS = int(
    os.getenv("CATEGORY_FILTER_MIN_RESULTS", "2")
)

# 分类检测器熔断参数
CATEGORY_CLASSIFIER_TIMEOUT = 8.0