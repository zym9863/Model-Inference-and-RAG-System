# Qwen3-8B 4bit推理与小上下文RAG系统

[English](./README-EN.md) | 中文

基于Qwen/Qwen3-8B模型的4bit量化推理系统，集成ChromaDB向量数据库和LlamaIndex框架，实现高效的检索增强生成(RAG)。

## 技术栈

- **LLM模型**: Qwen/Qwen3-8B (4bit量化)
- **嵌入模型**: google/embeddinggemma-300m
- **向量数据库**: ChromaDB
- **RAG框架**: LlamaIndex
- **量化技术**: bitsandbytes

## 功能特性

- 🚀 4bit量化推理，显存占用低
- 📚 基于ChromaDB的高效向量检索
- 🔍 LlamaIndex驱动的RAG管道
- 🎯 小上下文窗口优化
- ⚡ 快速响应和高吞吐量

## 项目结构

```
├── src/
│   ├── models/          # 模型加载和推理
│   ├── embeddings/      # 嵌入模型服务
│   ├── vectordb/        # ChromaDB向量数据库
│   ├── rag/             # RAG查询处理
│   └── utils/           # 工具函数
├── config/              # 配置文件
├── tests/               # 测试代码
├── examples/            # 使用示例
└── data/                # 数据文件
```

## 系统要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 至少8GB内存
- 至少10GB磁盘空间

## 安装

### Windows用户

1. 运行自动安装脚本：
```cmd
install.bat
```

2. 或手动安装：
```cmd
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate.bat

# 安装依赖
pip install -r requirements.txt
```

### Linux/Mac用户

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 交互式使用

```bash
# 启动交互式RAG系统
python main.py --mode interactive

# 或使用Windows脚本
run_example.bat
```

### 2. 编程接口

```python
from src.rag.query_processor import RAGQueryProcessor

# 初始化RAG系统
rag_processor = RAGQueryProcessor(
    llm_model_name="Qwen/Qwen3-8B",
    embedding_model_name="google/embeddinggemma-300m",
    use_llama_index=True
)

# 添加文档
documents = [
    {"text": "Python是一种编程语言", "metadata": {"topic": "编程"}},
    {"text": "机器学习是AI的分支", "metadata": {"topic": "AI"}}
]
rag_processor.add_documents(documents)

# 查询
result = rag_processor.query("什么是Python？")
print(f"回答: {result['response']}")
```

### 3. 命令行使用

```bash
# 单次查询
python main.py --mode batch --query "什么是人工智能？"

# 添加文档
python main.py --documents doc1.txt doc2.txt

# 显示系统信息
python main.py --info

# 清空文档
python main.py --clear
```

## 示例

### 运行基本示例
```bash
python examples/basic_usage.py
```

### 运行文档处理示例
```bash
python examples/document_processing.py
```

## 配置

系统配置文件位于 `config/config.yaml`：

```yaml
# 模型配置
model:
  llm_model_name: "Qwen/Qwen3-8B"
  embedding_model_name: "google/embeddinggemma-300m"
  device: "auto"

# RAG配置
rag:
  use_llama_index: true
  max_context_length: 1500
  top_k_retrieval: 5
  similarity_threshold: 0.7

# 生成配置
generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
```

### 环境变量配置

可以通过环境变量覆盖配置：

```bash
set RAG_SYSTEM_LLM_MODEL=your-model-name
set RAG_SYSTEM_DEVICE=cuda
set RAG_SYSTEM_PERSIST_DIRECTORY=./custom_data
```

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python tests/test_rag_system.py
```

## 性能优化

1. **GPU加速**: 设置 `device: "cuda"` 使用GPU
2. **内存优化**: 调整 `chunk_size` 和 `max_context_length`
3. **批处理**: 使用 `batch_size` 参数优化嵌入计算
4. **缓存**: 设置 `cache_dir` 缓存模型文件

## 故障排除

### 常见问题

1. **内存不足**: 减小 `chunk_size` 或使用CPU模式
2. **模型下载慢**: 设置HuggingFace镜像或使用本地模型
3. **CUDA错误**: 检查CUDA版本兼容性

### 日志查看

日志文件位于 `logs/rag_system.log`，可以查看详细的运行信息。

## 许可证

MIT License
