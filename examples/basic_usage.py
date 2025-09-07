"""
基本使用示例

演示RAG系统的基本功能，包括文档添加和查询。
"""

import sys
import os
from pathlib import Path

# 获取项目根目录并添加到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 确保工作目录是项目根目录
os.chdir(project_root)

from src.rag.query_processor import RAGQueryProcessor
from src.utils.logger import setup_logger


def main():
    """基本使用示例"""
    # 设置日志
    logger = setup_logger(level="INFO")
    
    print("=== RAG系统基本使用示例 ===\n")
    
    try:
        # 1. 初始化RAG系统
        print("1. 初始化RAG系统...")
        rag_processor = RAGQueryProcessor(
            llm_model_name="Qwen/Qwen3-8B",
            embedding_model_name="google/embeddinggemma-300m",
            use_llama_index=True,
            max_context_length=1000,
            top_k_retrieval=3
        )
        print("✓ RAG系统初始化完成\n")
        
        # 2. 准备示例文档
        print("2. 准备示例文档...")
        sample_documents = [
            {
                "text": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、问题解决、感知和语言理解。",
                "metadata": {"source": "ai_intro", "topic": "人工智能"}
            },
            {
                "text": "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。它基于算法和统计模型，使系统能够从数据中学习模式。",
                "metadata": {"source": "ml_intro", "topic": "机器学习"}
            },
            {
                "text": "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。它在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
                "metadata": {"source": "dl_intro", "topic": "深度学习"}
            },
            {
                "text": "自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。它包括文本分析、语言生成、机器翻译和情感分析等任务。",
                "metadata": {"source": "nlp_intro", "topic": "自然语言处理"}
            },
            {
                "text": "大语言模型（LLM）是基于Transformer架构的深度学习模型，能够理解和生成人类语言。它们通过在大量文本数据上进行预训练来学习语言模式。",
                "metadata": {"source": "llm_intro", "topic": "大语言模型"}
            }
        ]
        print(f"✓ 准备了 {len(sample_documents)} 个示例文档\n")
        
        # 3. 添加文档到RAG系统
        print("3. 添加文档到RAG系统...")
        doc_ids = rag_processor.add_documents(sample_documents, show_progress=True)
        print(f"✓ 成功添加 {len(doc_ids)} 个文档\n")
        
        # 4. 显示系统信息
        print("4. 系统信息:")
        system_info = rag_processor.get_system_info()
        for key, value in system_info.items():
            if not isinstance(value, dict):
                print(f"   {key}: {value}")
        print()
        
        # 5. 执行示例查询
        print("5. 执行示例查询...")
        
        queries = [
            "什么是人工智能？",
            "机器学习和深度学习有什么区别？",
            "大语言模型是如何工作的？",
            "自然语言处理有哪些应用？"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- 查询 {i}: {query} ---")
            
            # 执行查询
            result = rag_processor.query(
                query_text=query,
                max_new_tokens=256,
                temperature=0.7,
                include_sources=True
            )
            
            # 显示结果
            if "error" in result:
                print(f"❌ 查询失败: {result['error']}")
                continue
            
            print(f"回答: {result['response']}")
            
            # 显示源文档
            if "sources" in result and result["sources"]:
                print(f"\n参考文档 ({result['num_sources']} 个):")
                for j, source in enumerate(result["sources"], 1):
                    score = source.get("score", 0)
                    topic = source.get("metadata", {}).get("topic", "未知")
                    print(f"  {j}. {topic} (相似度: {score:.3f})")
            
            print(f"处理时间: {result.get('processing_time', 0):.2f} 秒")
        
        print("\n=== 示例完成 ===")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
