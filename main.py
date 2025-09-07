"""
RAG系统主程序入口

提供命令行界面和交互式查询功能。
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.rag.query_processor import RAGQueryProcessor


def main():
    """主程序入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="HuggingFace Qwen3-8B 4bit推理与小上下文RAG系统")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--mode", choices=["interactive", "batch", "server"], default="interactive", help="运行模式")
    parser.add_argument("--query", type=str, help="单次查询文本")
    parser.add_argument("--documents", type=str, nargs="+", help="要添加的文档文件路径")
    parser.add_argument("--clear", action="store_true", help="清空现有文档")
    parser.add_argument("--info", action="store_true", help="显示系统信息")
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = Config(args.config)
        config.apply_env_overrides()
        
        if not config.validate_config():
            print("配置验证失败，请检查配置文件")
            return 1
        
        # 设置日志
        logger = setup_logger(
            level=config.logging.level,
            log_file=config.logging.file_path,
            max_file_size=config.logging.max_file_size,
            backup_count=config.logging.backup_count
        )
        
        logger.info("RAG系统启动")
        
        # 初始化RAG处理器
        print("正在初始化RAG系统...")
        rag_processor = RAGQueryProcessor(
            llm_model_name=config.model.llm_model_name,
            embedding_model_name=config.model.embedding_model_name,
            chroma_persist_dir=config.vectordb.persist_directory,
            collection_name=config.vectordb.collection_name,
            use_llama_index=config.rag.use_llama_index,
            max_context_length=config.rag.max_context_length,
            top_k_retrieval=config.rag.top_k_retrieval,
            similarity_threshold=config.rag.similarity_threshold
        )
        print("RAG系统初始化完成！")
        
        # 处理命令行操作
        if args.clear:
            print("正在清空文档...")
            rag_processor.clear_documents()
            print("文档已清空")
        
        if args.documents:
            print(f"正在添加 {len(args.documents)} 个文档...")
            documents = load_documents(args.documents)
            rag_processor.add_documents(documents)
            print("文档添加完成")
        
        if args.info:
            show_system_info(rag_processor)
        
        # 运行模式
        if args.mode == "interactive":
            run_interactive_mode(rag_processor, config)
        elif args.mode == "batch":
            if args.query:
                result = rag_processor.query(args.query)
                print_query_result(result)
            else:
                print("批处理模式需要提供查询文本 (--query)")
                return 1
        elif args.mode == "server":
            run_server_mode(rag_processor, config)
        
        logger.info("RAG系统正常退出")
        return 0
        
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        return 0
    except Exception as e:
        print(f"系统错误: {e}")
        return 1


def load_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    加载文档文件
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        文档列表
    """
    documents = []
    
    for file_path in file_paths:
        try:
            path = Path(file_path)
            if not path.exists():
                print(f"警告: 文件不存在 {file_path}")
                continue
            
            # 读取文件内容
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:
                documents.append({
                    "text": content,
                    "metadata": {
                        "source": str(path),
                        "filename": path.name,
                        "size": len(content)
                    }
                })
                print(f"已加载: {path.name} ({len(content)} 字符)")
            
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
    
    return documents


def show_system_info(rag_processor: RAGQueryProcessor) -> None:
    """
    显示系统信息
    
    Args:
        rag_processor: RAG查询处理器
    """
    print("\n=== 系统信息 ===")
    
    info = rag_processor.get_system_info()
    
    print(f"语言模型: {info.get('llm_model', 'N/A')}")
    print(f"嵌入模型: {info.get('embedding_model', 'N/A')}")
    print(f"使用LlamaIndex: {info.get('use_llama_index', 'N/A')}")
    print(f"集合名称: {info.get('collection_name', 'N/A')}")
    print(f"最大上下文长度: {info.get('max_context_length', 'N/A')}")
    print(f"检索文档数量: {info.get('top_k_retrieval', 'N/A')}")
    print(f"相似度阈值: {info.get('similarity_threshold', 'N/A')}")
    
    # 向量数据库信息
    if "vector_db_info" in info:
        db_info = info["vector_db_info"]
        print(f"文档数量: {db_info.get('count', 'N/A')}")
    
    print()


def run_interactive_mode(rag_processor: RAGQueryProcessor, config: Config) -> None:
    """
    运行交互式模式
    
    Args:
        rag_processor: RAG查询处理器
        config: 系统配置
    """
    print("\n=== 交互式RAG查询系统 ===")
    print("输入查询文本，输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助信息")
    print("输入 'info' 查看系统信息")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n请输入查询: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if query.lower() == 'help':
                show_help()
                continue
            
            if query.lower() == 'info':
                show_system_info(rag_processor)
                continue
            
            # 执行查询
            print("正在处理查询...")
            result = rag_processor.query(
                query,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature
            )
            
            print_query_result(result)
            
        except KeyboardInterrupt:
            print("\n用户中断")
            break
        except Exception as e:
            print(f"查询处理错误: {e}")


def show_help() -> None:
    """显示帮助信息"""
    print("\n=== 帮助信息 ===")
    print("可用命令:")
    print("  help  - 显示此帮助信息")
    print("  info  - 显示系统信息")
    print("  quit  - 退出系统")
    print("  exit  - 退出系统")
    print("  q     - 退出系统")
    print("\n直接输入文本进行查询")


def print_query_result(result: Dict[str, Any]) -> None:
    """
    打印查询结果
    
    Args:
        result: 查询结果字典
    """
    print("\n" + "=" * 50)
    print("查询结果:")
    print("-" * 50)
    
    if "error" in result:
        print(f"错误: {result['error']}")
        return
    
    print(f"问题: {result.get('query', 'N/A')}")
    print(f"回答: {result.get('response', 'N/A')}")
    
    # 显示源文档信息
    if "sources" in result and result["sources"]:
        print(f"\n参考文档 ({result.get('num_sources', 0)} 个):")
        for i, source in enumerate(result["sources"][:3], 1):  # 只显示前3个
            score = source.get("score", 0)
            text = source.get("text", "")[:100] + "..." if len(source.get("text", "")) > 100 else source.get("text", "")
            print(f"  {i}. 相似度: {score:.3f}")
            print(f"     内容: {text}")
    
    # 显示性能信息
    if "processing_time" in result:
        print(f"\n处理时间: {result['processing_time']:.2f} 秒")
    
    print("=" * 50)


def run_server_mode(rag_processor: RAGQueryProcessor, config: Config) -> None:
    """
    运行服务器模式
    
    Args:
        rag_processor: RAG查询处理器
        config: 系统配置
    """
    print("服务器模式暂未实现")
    print("请使用交互式模式或批处理模式")


if __name__ == "__main__":
    exit(main())
