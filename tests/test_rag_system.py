"""
RAG系统单元测试

测试RAG系统的各个组件功能。
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.qwen3_model import Qwen3Model
from src.embeddings.embedding_service import EmbeddingService
from src.vectordb.chromadb_manager import ChromaDBManager
from src.rag.query_processor import RAGQueryProcessor
from src.utils.config import Config


class TestEmbeddingService(unittest.TestCase):
    """测试嵌入服务"""
    
    def setUp(self):
        """测试前设置"""
        self.embedding_service = EmbeddingService(
            model_name="google/gemma-2b",
            device="cpu"  # 使用CPU避免GPU依赖
        )
    
    def test_encode_single_text(self):
        """测试单个文本编码"""
        text = "这是一个测试文本"
        embeddings = self.embedding_service.encode_text(text)
        
        self.assertEqual(embeddings.shape[0], 1)
        self.assertGreater(embeddings.shape[1], 0)
    
    def test_encode_multiple_texts(self):
        """测试多个文本编码"""
        texts = ["文本1", "文本2", "文本3"]
        embeddings = self.embedding_service.encode_text(texts)
        
        self.assertEqual(embeddings.shape[0], len(texts))
        self.assertGreater(embeddings.shape[1], 0)
    
    def test_similarity_computation(self):
        """测试相似度计算"""
        texts1 = ["人工智能", "机器学习"]
        texts2 = ["AI技术", "深度学习"]
        
        embeddings1 = self.embedding_service.encode_text(texts1)
        embeddings2 = self.embedding_service.encode_text(texts2)
        
        similarity = self.embedding_service.compute_similarity(embeddings1, embeddings2)
        
        self.assertEqual(similarity.shape, (len(texts1), len(texts2)))
        self.assertTrue((similarity >= -1).all() and (similarity <= 1).all())


class TestChromaDBManager(unittest.TestCase):
    """测试ChromaDB管理器"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = ChromaDBManager(
            persist_directory=self.temp_dir,
            collection_name="test_collection"
        )
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_documents(self):
        """测试添加文档"""
        documents = ["文档1", "文档2", "文档3"]
        metadatas = [{"source": f"doc_{i}"} for i in range(len(documents))]
        
        doc_ids = self.db_manager.add_documents(
            documents=documents,
            metadatas=metadatas
        )
        
        self.assertEqual(len(doc_ids), len(documents))
        
        # 检查集合信息
        info = self.db_manager.get_collection_info()
        self.assertEqual(info["count"], len(documents))
    
    def test_query_documents(self):
        """测试查询文档"""
        # 先添加文档
        documents = ["人工智能技术", "机器学习算法", "深度学习网络"]
        self.db_manager.add_documents(documents=documents)
        
        # 查询
        results = self.db_manager.query_documents(
            query_texts="AI技术",
            n_results=2
        )
        
        self.assertIn("documents", results)
        self.assertLessEqual(len(results["documents"][0]), 2)
    
    def test_delete_documents(self):
        """测试删除文档"""
        documents = ["文档1", "文档2"]
        doc_ids = self.db_manager.add_documents(documents=documents)
        
        # 删除第一个文档
        self.db_manager.delete_documents(ids=[doc_ids[0]])
        
        # 检查剩余文档数量
        info = self.db_manager.get_collection_info()
        self.assertEqual(info["count"], 1)


class TestRAGQueryProcessor(unittest.TestCase):
    """测试RAG查询处理器"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 使用较小的配置进行测试
        self.rag_processor = RAGQueryProcessor(
            llm_model_name="Qwen/Qwen2.5-7B-Instruct",
            embedding_model_name="google/gemma-2b",
            chroma_persist_dir=self.temp_dir,
            use_llama_index=False,  # 使用自定义组件便于测试
            max_context_length=500,
            top_k_retrieval=2
        )
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_documents(self):
        """测试添加文档"""
        documents = [
            {"text": "Python是一种编程语言", "metadata": {"topic": "编程"}},
            {"text": "机器学习是AI的分支", "metadata": {"topic": "AI"}},
            {"text": "数据科学处理大数据", "metadata": {"topic": "数据"}}
        ]
        
        doc_ids = self.rag_processor.add_documents(documents)
        self.assertEqual(len(doc_ids), len(documents))
    
    def test_query_processing(self):
        """测试查询处理"""
        # 添加测试文档
        documents = [
            "Python是一种高级编程语言，广泛用于数据科学和AI开发",
            "机器学习是人工智能的核心技术，能够从数据中学习模式",
            "深度学习使用神经网络来解决复杂问题"
        ]
        
        self.rag_processor.add_documents(documents)
        
        # 执行查询
        result = self.rag_processor.query(
            query_text="什么是Python？",
            max_new_tokens=100,
            include_sources=True
        )
        
        self.assertIn("response", result)
        self.assertIn("query", result)
        self.assertNotIn("error", result)
    
    def test_system_info(self):
        """测试系统信息获取"""
        info = self.rag_processor.get_system_info()
        
        self.assertIn("llm_model", info)
        self.assertIn("embedding_model", info)
        self.assertIn("use_llama_index", info)


class TestConfig(unittest.TestCase):
    """测试配置管理"""
    
    def setUp(self):
        """测试前设置"""
        self.temp_config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.temp_config_file.close()
    
    def tearDown(self):
        """测试后清理"""
        Path(self.temp_config_file.name).unlink(missing_ok=True)
    
    def test_default_config(self):
        """测试默认配置"""
        config = Config(self.temp_config_file.name)
        
        self.assertIsNotNone(config.model.llm_model_name)
        self.assertIsNotNone(config.model.embedding_model_name)
        self.assertGreater(config.rag.max_context_length, 0)
    
    def test_config_validation(self):
        """测试配置验证"""
        config = Config(self.temp_config_file.name)
        
        # 默认配置应该有效
        self.assertTrue(config.validate_config())
        
        # 修改为无效配置
        config.rag.max_context_length = -1
        self.assertFalse(config.validate_config())
    
    def test_config_update(self):
        """测试配置更新"""
        config = Config(self.temp_config_file.name)
        
        original_value = config.rag.top_k_retrieval
        new_value = original_value + 1
        
        config.update_config("rag", top_k_retrieval=new_value)
        self.assertEqual(config.rag.top_k_retrieval, new_value)


def run_performance_test():
    """运行性能测试"""
    print("\n=== 性能测试 ===")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 初始化系统
        rag_processor = RAGQueryProcessor(
            chroma_persist_dir=temp_dir,
            use_llama_index=False,
            max_context_length=1000
        )
        
        # 准备测试数据
        documents = [
            f"这是测试文档 {i}，包含一些关于人工智能和机器学习的内容。"
            for i in range(10)
        ]
        
        # 测试文档添加性能
        import time
        start_time = time.time()
        rag_processor.add_documents(documents)
        add_time = time.time() - start_time
        
        print(f"添加 {len(documents)} 个文档耗时: {add_time:.2f} 秒")
        
        # 测试查询性能
        queries = [
            "什么是人工智能？",
            "机器学习的应用有哪些？",
            "深度学习和传统机器学习的区别？"
        ]
        
        total_query_time = 0
        for query in queries:
            start_time = time.time()
            result = rag_processor.query(query, max_new_tokens=100)
            query_time = time.time() - start_time
            total_query_time += query_time
            
            print(f"查询 '{query[:20]}...' 耗时: {query_time:.2f} 秒")
        
        avg_query_time = total_query_time / len(queries)
        print(f"平均查询时间: {avg_query_time:.2f} 秒")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # 运行单元测试
    print("=== 运行单元测试 ===")
    unittest.main(verbosity=2, exit=False)
    
    # 运行性能测试
    run_performance_test()
