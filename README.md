# 智医助手 · 流行病学知识问答智能体

> 基于 LangChain + RAG + Agent 的中文医学知识问答系统
>
> 检索深度动态调度 · 事实锚定 · 反幻觉 · 熔断降级 · Token 预算控制 · Web 界面

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/langchain-0.3+-green.svg)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 📖 目录

- [项目简介](#-项目简介)
- [核心特性](#-核心特性)
- [系统架构](#-系统架构)
- [目录结构](#-目录结构)
- [快速开始](#-快速开始)
- [配置说明](#-配置说明)
- [数据集准备](#-数据集准备)
- [使用方式](#-使用方式)
- [测试](#-测试)
- [常见问题](#-常见问题)
- [开发说明](#-开发说明)
- [许可证](#-许可证)
- [致谢](#-致谢)

---

## 🎯 项目简介

**智医助手** 是一个面向中文医学知识问答的智能体系统。它结合了 **RAG（检索增强生成）** 与 **Agent（工具调用）** 两种模式，通过动态判断问题复杂度来选择检索深度，并用严格的**事实锚定机制**避免大模型幻觉。

**适合场景**：

- 流行病学 / 常见疾病 / 药物知识的科普问答
- 医学教材、指南、百科类文档的智能检索
- 需要引用来源、拒绝编造的严肃知识问答

**不适合场景**：

- ❌ 线上诊断、开处方、用药剂量咨询（系统会主动拒绝）
- ❌ 急诊分诊（系统会提示拨打 120）

---

## ✨ 核心特性

### 🔍 检索层

- **检索深度动态调度**：LLM 判断问题复杂度，自动选择三档深度
  - `basic` —— 简单名词解释，纯向量检索
  - `standard` —— 一般医学咨询，BM25 + 向量双路召回
  - `deep` —— 复杂/危重问题，HyDE + BM25 + Reranker + 父文档检索
- **父文档检索**：子块（300 字）用于召回，父块（1200 字）返回给 LLM，兼顾精度与上下文完整性
- **HyDE 查询增强**：用 LLM 生成假设答案再做向量检索，显著提升口语化问题的召回率
- **分类元数据过滤**（可选）：基于数据 `categories` 字段过滤检索范围，命中率低时自动放宽
- **多路召回融合**：BM25 稀疏 + 向量稠密，RRF 融合

### 🛡️ 安全层

- **统一安全检测**：`utils/safety.py` 集中处理
  - 紧急症状（胸痛 / 大出血 / 昏迷…）→ 立即建议拨打 120
  - 个人诊疗请求（开药 / 诊断 / 剂量…）→ 拒绝并建议线下就医
  - 提示注入 / 越狱攻击（"忽略以上指令" / "DAN 模式"…）→ 拦截
  - 知识提问白名单 → 放行（阈值放宽到 0.8）
- **事实锚定**：最终答案必须来自检索原文，否则强制拒绝
- **反幻觉检测**：识别"承认找不到 + 又编造"模式，直接短路
- **引用强制**：每个医学事实必须带【来源 N】标记

### ⚙️ 工程层

- **熔断器 + 慢调用熔断**：`utils/circuit_breaker.py`，独立管理每个外部依赖
- **多层降级链**：主检索 → 纯向量 → BM25 → 静态兜底，任一环节挂掉不影响整体
- **Token 三级预算**：单请求 / 单会话 / 全局，超限抛出 `TokenBudgetExceeded`
- **Embedding 缓存**：线程安全 LRU，避免重复计算
- **LLM 缓存**：支持内存 / Redis（失败自动降级到内存）
- **文件热更新**：watchdog 监听数据目录，新增文件自动入库（增量更新 FAISS + BM25 + 父文档索引）

### 🌐 接口层

- **CLI 模式**：`python main.py` 交互式问答
- **Web 模式**：FastAPI + 原生前端，开箱即用

---

## 🏗️ 系统架构

```
                          用户提问
                             │
                             ▼
                  ┌──────────────────────┐
                  │  1. 统一安全检测     │
                  │  utils/safety.py     │
                  └──────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        紧急症状        攻击/注入      正常问题
        →拨打120       →拦截          │
                                        ▼
                          ┌────────────────────────┐
                          │ 2. 深度分类器          │
                          │ LLM → basic/std/deep   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │ 3. 分类检测器（可选）  │
                          │ LLM → 内科 / 药学 / …  │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │ 4. 检索器组装          │
                          │ core/tools/deep_search │
                          └───────────┬────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
    [deep]                       [standard]                      [basic]
  HyDE + BM25                  BM25 + 向量                     纯向量
  + Reranker                   + 分类过滤                      + 分类过滤
  + 父文档检索
        │                             │                             │
        └─────────────┬───────────────┴─────────────────────────────┘
                      ▼
          ┌───────────────────────────┐
          │ 5. 熔断 + 降级链          │
          │ 主检索→向量→BM25→兜底    │
          └───────────┬───────────────┘
                      │
                      ▼
          ┌───────────────────────────┐
          │ 6. Agent 生成 + 事实锚定  │
          │ agents/medical_agent.py   │
          └───────────┬───────────────┘
                      │
                      ▼
          ┌───────────────────────────┐
          │ 7. 落库（QA Chroma）      │
          └───────────┬───────────────┘
                      │
                      ▼
                带【来源 N】的回答
```

---

## 📁 目录结构

```
.
├── agents/                     # Agent 定义与执行
│   └── medical_agent.py        # 医学 Agent：工具调用 + 事实锚定
├── core/
│   ├── cache/                  # 缓存
│   │   └── cache_manager.py    # LLM 缓存 + Embedding 缓存
│   ├── embeddings/             # 向量化
│   │   └── embedding_factory.py
│   ├── loaders/                # 文档加载
│   │   └── document_loader.py
│   ├── retrievers/             # 检索器
│   │   ├── base_retriever.py        # BM25 / 向量 / Ensemble
│   │   ├── hyde.py                  # HyDE 检索
│   │   ├── parent_doc_retriever.py  # 父文档检索
│   │   └── rerank.py                # Cross-Encoder 重排
│   ├── splitters/              # 文本切分
│   │   └── splitter.py
│   ├── tools/                  # 检索组装
│   │   └── deep_search.py      # 深度组装器
│   └── vector_stores/          # 向量库
│       ├── faiss_store.py      # 主知识库
│       └── qa_chroma.py        # QA 历史
├── models/
│   └── llm.py                  # LLM 入口（带熔断）
├── tools/
│   └── search_tools.py         # Tool 层：深度分类 + 检索调度
├── utils/
│   ├── circuit_breaker.py      # 熔断器
│   ├── file_watcher.py         # 文件监控
│   ├── safety.py               # 统一安全检测
│   ├── text_cleaner.py         # 文本清洗
│   └── token_tracker.py        # Token 统计
├── static/                     # Web 前端资源
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/                      # 测试
│   ├── test_fixes.py           # 单元测试
│   ├── test_faiss_filter.py    # 集成测试
│   └── test_e2e.py             # 端到端测试
├── data/
│   └── medical_knowledge/      # 医学知识文档（不提交 Git）
├── config.py                   # 配置
├── main.py                     # CLI 入口
├── web_app.py                  # Web 服务入口
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 快速开始

### 环境要求

| 项目 | 版本 |
|---|---|
| Python | 3.10+ |
| 内存 | ≥ 8GB（含 Embedding 模型） |
| 显存 | 可选，有 GPU 更快（≥ 6GB 建议） |
| 磁盘 | ≥ 5GB（含模型 + 索引） |

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> 💡 如果安装 `unstructured` 或 `pdf2image` 失败，可以参考官方文档安装系统依赖：
> - Windows: 安装 [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) 和 [Poppler](https://github.com/oschwartz10612/poppler-windows/releases)
> - Linux: `sudo apt install tesseract-ocr poppler-utils`

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，**至少**填写以下两项：

```env
# 智谱 API Key（必填）
ZHIPU_API_KEY=xxxxxxxx.yyyyyyyy.zzzzzzzz

# 本机模型路径（按实际修改）
EMBEDDING_MODEL=D:/model/m3e-large
```

### 5. 准备模型

需要下载以下模型到本机（或修改 `.env` 指向已有路径）：

| 模型 | 用途 | 是否必须 | 大小 |
|---|---|---|---|
| [m3e-large](https://huggingface.co/moka-ai/m3e-large) | 中文 Embedding | ✅ 必须 | ~1.3GB |
| [bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | 重排序（deep 档用） | ⚠️ 建议 | ~1.3GB |
| [Unified_Prompt_Guard](https://huggingface.co/) | 提示注入检测 | ⚪ 可选 | ~500MB |

**不装安全模型** → 设 `SAFETY_MODEL_ENABLED=false`（仅规则过滤，效果也不错）

### 6. 准备数据

将医学知识文档放入 `data/medical_knowledge/`，支持格式：

- `.txt` / `.md` / `.markdown`
- `.pdf`（扫描版自动 OCR）
- `.docx` / `.doc`
- `.csv`
- `.jsonl`（**推荐**，见[数据集准备](#-数据集准备)）
- `.jpg` / `.png` / `.jpeg` / `.bmp` / `.tiff`（图片 OCR）

### 7. 首次启动

```bash
# CLI 模式（首次会构建向量库，需要较长时间）
python main.py --rebuild

# 或者 Web 模式
python web_app.py
# 浏览器打开 http://localhost:8000
```

---

## ⚙️ 配置说明

所有配置集中在 `config.py` + `.env`。常用开关：

### API 与模型

| 环境变量 | 说明 | 默认 |
|---|---|---|
| `ZHIPU_API_KEY` | 智谱 API Key | 必填 |
| `ZHIPU_API_BASE` | API 地址 | `https://open.bigmodel.cn/api/paas/v4/` |
| `DEFAULT_MODEL` | 主 LLM | `glm-4-air` |
| `EMBEDDING_MODEL` | Embedding 模型路径 | `D:/model/m3e-large` |
| `RERANKER_MODEL` | Reranker 模型路径 | `D:/model/bge-reranker-large` |
| `SAFETY_MODEL` | 安全模型路径 | `D:/model/Unified_Prompt_Guard` |
| `SAFETY_MODEL_ENABLED` | 是否启用安全模型 | `true` |

### 检索深度

见 `config.py` 的 `RETRIEVAL_DEPTH_CONFIG`：

| 深度 | top_k | HyDE | BM25 | Reranker | 父文档 |
|---|---|---|---|---|---|
| `basic` | 2 | ❌ | ✅ | ❌ | ❌ |
| `standard` | 4 | ❌ | ✅ | ❌ | ❌ |
| `deep` | 5 | ✅ | ✅ | ✅ | ✅ |

### 分类过滤（可选）

| 环境变量 | 说明 | 默认 |
|---|---|---|
| `ENABLE_CATEGORY_FILTER` | 是否启用分类元数据过滤 | `true` |
| `CATEGORY_FILTER_RECALL_MULTIPLIER` | 过滤前多召回倍数 | `3` |
| `CATEGORY_FILTER_MIN_RESULTS` | 过滤后少于此值则放宽 | `2` |

### Token 预算

| 环境变量 | 说明 | 默认 |
|---|---|---|
| `TOKEN_BUDGET_PER_SESSION` | 单会话上限 | `100000` |
| `TOKEN_BUDGET_TOTAL` | 全局上限 | `1000000` |
| `TOKEN_WARN_RATIO` | 预警比例 | `0.8` |

### 缓存

| 环境变量 | 说明 | 默认 |
|---|---|---|
| `CACHE_ENABLED` | 是否启用缓存 | `true` |
| `LLM_CACHE_BACKEND` | `memory` / `redis` | `memory` |
| `REDIS_URL` | Redis 地址 | `redis://localhost:6379/0` |
| `EMBEDDING_CACHE_SIZE` | Embedding 缓存条数 | `10000` |

---

## 📊 数据集准备

本项目适配 **A Hospital Medical Wiki Dataset**（A+ 医学百科），格式为 JSONL：

```json
{
  "title": "高血压",
  "url": "http://www.a-hospital.com/w/高血压",
  "content": "高血压是指以体循环动脉血压...",
  "categories": ["内科", "心血管内科"],
  "content_length": 1190,
  "language": "zh-CN"
}
```

### 下载

数据集约 245MB，**不建议直接提交到 GitHub**。推荐通过以下方式获取：

- Hugging Face Datasets（搜索 `a-hospital-medical-wiki`）
- 或联系数据集作者

下载后解压到 `data/medical_knowledge/articles.jsonl`。

### 快速试用（推荐）

仓库内提供了 `data/medical_knowledge/sample.jsonl`（约 100 条），clone 后即可直接体验：

```bash
python main.py --rebuild
```

### 全量导入

245MB 数据首次构建 FAISS 需要较长时间（取决于 Embedding 速度）：

| 硬件 | 预计耗时 |
|---|---|
| CPU | 数小时 |
| RTX 3060 | 约 40 分钟 |
| RTX 4090 | 约 15 分钟 |

**建议**：先用 sample 验证流程，确认无误再导入全量。

---

## 💻 使用方式

### CLI 模式

```bash
python main.py                # 正常启动（含文件监控）
python main.py --rebuild      # 强制重建 FAISS 知识库
python main.py --no-cache     # 禁用缓存
python main.py --no-watch     # 禁用文件自动监控
python main.py --verbose      # 输出详细日志
python main.py --help         # 显示帮助
```

进入交互后：

```
❓ 您: 高血压的诊断标准是什么？
🤖 助手: 根据检索到的资料...
(⏱️ 耗时: 3.21 秒)
(🪙 本次消耗: 1234 tokens | 输入 890 / 输出 344)
```

**特殊命令**：

- 输入 `quit` / `exit` / `q` 退出
- 输入 `reset` 重置会话 Token 预算

### Web 模式

```bash
python web_app.py
# 或
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

浏览器访问 `http://localhost:8000`。

**API 接口**：

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/chat` | 提问 |
| `GET` | `/api/status` | 查询 Token 消耗 |
| `POST` | `/api/reset` | 重置会话预算 |

**请求示例**：

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "高血压的诊断标准是什么？"}'
```

---

## 🧪 测试

### 单元测试（不消耗 Token）

```bash
python tests/test_fixes.py
```

验证：
- FAISS `filter` 回调收到的是 `metadata` dict（不是 `Document`）
- 医学幻觉检测不误杀正常回答
- `_finalize` 不因中文分词问题误判

### 集成测试（需要已构建索引）

```bash
python tests/test_faiss_filter.py
```

验证：
- 无过滤检索 vs 分类过滤检索的结果差异
- 不存在的分类自动放宽
- 过滤后结果必须命中目标分类

### 端到端测试（消耗 Token）

```bash
python tests/test_e2e.py
```

覆盖 6 类用例：
- 知识提问 → 正常回答
- 复杂问题 → 深度检索
- 个人诊疗 → 拦截
- 紧急症状 → 拦截
- 提示注入 → 拦截
- 药物知识 → 正常回答

---

## ❓ 常见问题

### 🔴 401 认证失败

```
OpenAIAuthenticationError: Error code: 401 - 令牌已过期或验证不正确
```

**排查步骤**：

1. **Key 是否过期**：登录 [智谱开放平台](https://open.bigmodel.cn/)，看 API Key 有效期（免费 Key 一般 1 个月）
2. **`.env` 格式**：不要用引号，不要有前后空格
   ```env
   ZHIPU_API_KEY=xxxxxxxx.yyyyyyyy.zzzzzzzz    # ✅ 正确
   ZHIPU_API_KEY="xxx.yyy.zzz"                 # ❌ 引号会被保留
   ZHIPU_API_KEY=xxx.yyy.zzz                   # ❌ 末尾有空格
   ```
3. **验证 Key 读取**：
   ```bash
   python -c "from config import ZHIPU_API_KEY; print(repr(ZHIPU_API_KEY))"
   ```
   期望输出：`'xxxxxxxx.yyyyyyyy.zzzzzzzz'`

4. **网络 / 代理**：如果挂了梯子，可能被智谱认为是境外 IP，关掉再试

### 🔴 `ModuleNotFoundError: No module named 'xxx_loader'`

Python 3 不支持裸导入同级模块，需要**加前导点**：

```python
from basic_loader import ...    # ❌ 错误
from .basic_loader import ...   # ✅ 正确（文件同级）
```

### 🔴 `'dict' object has no attribute 'metadata'`

FAISS 的 `filter` 回调接收的是 **metadata 字典**，不是 `Document`：

```python
def _filter(metadata: dict) -> bool:   # ✅ 参数是 dict
    cats = metadata.get("categories") or []
    return bool(target & set(cats))
```

### 🔴 幻觉误判：`检测到幻觉术语: ['包括冠心病', ...]`

这是旧版正则分词不准导致的误杀。已在新版本移除该逻辑，确保 `medical_agent.py` 中：
- 删除了 `_MEDICAL_TERM`
- 删除了 `_ANCHOR_STOPWORDS`
- `_finalize` 不再有第 5 步医学实体校验

### 🔴 首次构建 FAISS 太慢

- 确认有 GPU：`python -c "import torch; print(torch.cuda.is_available())"`
- 换更小的 Embedding 模型（`m3e-base` 比 `m3e-large` 快 3 倍）
- 先用 `sample.jsonl` 验证流程

### 🔴 `Address already in use`

8000 端口被占，换端口：

```bash
python -m uvicorn web_app:app --port 8001
```

或查杀占用：

```powershell
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

```bash
# Linux / macOS
lsof -i :8000
kill -9 <PID>
```

### 🔴 文件监控不工作

```bash
pip install watchdog
```

或启动时加 `--no-watch` 跳过。

---

## 🛠️ 开发说明

### 代码风格

- 分层清晰：`core/` 纯逻辑，`tools/` 工具封装，`agents/` 编排，`utils/` 通用
- 熔断独立：每个外部依赖（LLM / Embedding / Redis / Reranker / FAISS）都有独立熔断器
- 降级优先：任一环节失败都要有兜底路径，不让主流程崩溃

### 扩展新的检索器

1. 在 `core/retrievers/` 下新建文件，实现 `BaseRetriever` 接口
2. 在 `core/tools/deep_search.py` 的 `_build_traditional_retriever` 中按开关挂载
3. 在 `config.RETRIEVAL_DEPTH_CONFIG` 里加对应的开关字段

### 扩展安全规则

编辑 `utils/safety.py` 顶部的关键词列表：

- `EMERGENCY_KEYWORDS` —— 紧急症状
- `DIAGNOSIS_KEYWORDS` —— 个人诊疗
- `ATTACK_KEYWORDS` —— 注入攻击
- `KNOWLEDGE_QUESTION_PATTERNS` —— 知识提问白名单

### 添加新工具

1. 在 `tools/search_tools.py` 里用 `StructuredTool.from_function` 定义
2. 加入 `ALL_TOOLS` 列表
3. Agent 会自动识别并调用

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 🙏 致谢

- 数据集来源：[A+医学百科](http://www.a-hospital.com/)
- 检索框架：[LangChain](https://github.com/langchain-ai/langchain)
- 向量库：[FAISS](https://github.com/facebookresearch/faiss) + [Chroma](https://github.com/chroma-core/chroma)
- Embedding：[m3e-large](https://huggingface.co/moka-ai/m3e-large)
- Reranker：[bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
- 大模型：[智谱 AI](https://open.bigmodel.cn/)

## ⚠️ 免责声明

本项目仅供**学习研究**使用。所有回答均为**知识科普**，**不构成诊疗建议**。
如有身体不适，请前往正规医院就诊。

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/wuyuhao545">wuyuhao545</a>
</p>