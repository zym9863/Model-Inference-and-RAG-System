"""
文档处理示例

演示如何处理不同类型的文档，包括文本文件、PDF等。
"""

import sys
from pathlib import Path
import json

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.rag.query_processor import RAGQueryProcessor
from src.utils.logger import setup_logger


def create_sample_documents():
    """创建示例文档文件"""
    data_dir = Path("data/sample_docs")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例文档
    documents = {
        "python_basics.txt": """
Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。
它支持多种编程范式，包括面向对象、函数式和过程式编程。
Python广泛应用于Web开发、数据科学、人工智能、自动化脚本等领域。

主要特点：
1. 语法简洁易读
2. 丰富的标准库
3. 强大的第三方生态系统
4. 跨平台兼容性
5. 解释型语言，开发效率高
        """,
        
        "machine_learning.txt": """
机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习模式，
而无需明确编程每个决策规则。

主要类型：
1. 监督学习：使用标记数据训练模型
2. 无监督学习：从未标记数据中发现模式
3. 强化学习：通过与环境交互学习最优策略

常见算法：
- 线性回归
- 决策树
- 随机森林
- 支持向量机
- 神经网络
        """,
        
        "data_science.txt": """
数据科学是一个跨学科领域，结合了统计学、计算机科学和领域专业知识，
从数据中提取有价值的洞察。

数据科学流程：
1. 数据收集：从各种来源获取数据
2. 数据清洗：处理缺失值、异常值和不一致性
3. 探索性数据分析：理解数据的分布和关系
4. 特征工程：创建和选择有用的特征
5. 模型构建：选择和训练适当的模型
6. 模型评估：验证模型性能
7. 部署和监控：将模型投入生产使用

常用工具：
- Python/R：编程语言
- Pandas：数据处理
- NumPy：数值计算
- Matplotlib/Seaborn：数据可视化
- Scikit-learn：机器学习
- TensorFlow/PyTorch：深度学习
        """
    }
    
    # 写入文件
    for filename, content in documents.items():
        file_path = data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"创建文档: {file_path}")
    
    return list(data_dir.glob("*.txt"))


def load_documents_from_files(file_paths):
    """从文件加载文档"""
    documents = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:
                documents.append({
                    "text": content,
                    "metadata": {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "size": len(content),
                        "type": "text_file"
                    }
                })
                print(f"加载文档: {file_path.name} ({len(content)} 字符)")
        
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
    
    return documents


def main():
    """文档处理示例"""
    # 设置日志
    logger = setup_logger(level="INFO")
    
    print("=== 文档处理示例 ===\n")
    
    try:
        # 1. 创建示例文档
        print("1. 创建示例文档...")
        file_paths = create_sample_documents()
        print(f"✓ 创建了 {len(file_paths)} 个示例文档\n")
        
        # 2. 初始化RAG系统
        print("2. 初始化RAG系统...")
        rag_processor = RAGQueryProcessor(
            use_llama_index=True,
            max_context_length=1500,
            top_k_retrieval=3,
            similarity_threshold=0.6
        )
        print("✓ RAG系统初始化完成\n")
        
        # 3. 加载文档
        print("3. 从文件加载文档...")
        documents = load_documents_from_files(file_paths)
        print(f"✓ 成功加载 {len(documents)} 个文档\n")
        
        # 4. 添加文档到RAG系统
        print("4. 添加文档到RAG系统...")
        doc_ids = rag_processor.add_documents(documents, show_progress=True)
        print(f"✓ 成功添加 {len(doc_ids)} 个文档到RAG系统\n")
        
        # 5. 文档查询测试
        print("5. 执行文档查询测试...")
        
        test_queries = [
            "Python有哪些主要特点？",
            "机器学习有哪些类型？",
            "数据科学的流程是什么？",
            "数据科学常用的工具有哪些？",
            "什么是监督学习和无监督学习？"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 查询 {i}: {query} ---")
            
            result = rag_processor.query(
                query_text=query,
                max_new_tokens=300,
                temperature=0.7,
                include_sources=True
            )
            
            if "error" in result:
                print(f"❌ 查询失败: {result['error']}")
                continue
            
            print(f"回答: {result['response']}")
            
            # 显示源文档信息
            if "sources" in result and result["sources"]:
                print(f"\n参考文档:")
                for j, source in enumerate(result["sources"], 1):
                    metadata = source.get("metadata", {})
                    filename = metadata.get("filename", "未知文件")
                    score = source.get("score", 0)
                    print(f"  {j}. {filename} (相似度: {score:.3f})")
            
            print(f"处理时间: {result.get('processing_time', 0):.2f} 秒")
        
        # 6. 保存查询结果
        print("\n6. 保存查询结果...")
        results_file = Path("data/query_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        for query in test_queries:
            result = rag_processor.query(query, include_sources=True)
            all_results.append({
                "query": query,
                "response": result.get("response", ""),
                "num_sources": result.get("num_sources", 0),
                "processing_time": result.get("processing_time", 0)
            })
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 查询结果已保存到: {results_file}")
        
        # 7. 系统统计信息
        print("\n7. 系统统计信息:")
        system_info = rag_processor.get_system_info()
        print(f"   文档数量: {system_info.get('vector_db_info', {}).get('count', 'N/A')}")
        print(f"   集合名称: {system_info.get('collection_name', 'N/A')}")
        print(f"   检索方法: {'LlamaIndex' if system_info.get('use_llama_index') else '自定义'}")
        
        print("\n=== 文档处理示例完成 ===")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
