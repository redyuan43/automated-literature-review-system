# step2_ollama.py - 使用Ollama的文献报告生成模块

import json
import os
import logging
import sys
from datetime import datetime
from typing import Dict, Optional, List
import re
import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document

# 导入我们的Ollama客户端
from ollama_client import generate_text, chat_with_model, ollama_client
from config import MODEL_CONFIG, SYSTEM_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('step2_ollama.log')
    ]
)
logger = logging.getLogger(__name__)

class OllamaLiteratureProcessor:
    """基于Ollama的文献处理器"""
    
    def __init__(self):
        self.main_model = MODEL_CONFIG.main_model
        self.embeddings_dir = SYSTEM_CONFIG.embeddings_dir
        self.output_dir = SYSTEM_CONFIG.output_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"初始化Ollama文献处理器，主模型: {self.main_model}")
        
        # 验证Ollama连接
        if not ollama_client.health_check():
            raise ConnectionError("无法连接到Ollama服务器")
        
        # 检查模型是否可用
        available_models = [m['name'] for m in ollama_client.list_models()]
        if self.main_model not in available_models:
            logger.warning(f"模型 {self.main_model} 不可用，可用模型: {available_models}")
            if available_models:
                self.main_model = available_models[0]
                logger.info(f"使用备用模型: {self.main_model}")
            else:
                raise RuntimeError("没有可用的模型")
    
    def load_vector_store(self) -> FAISS:
        """加载向量数据库"""
        try:
            logger.info(f"从 {self.embeddings_dir} 加载向量数据库...")
            
            # 创建嵌入模型
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}  # 使用CPU进行嵌入
            )
            
            # 检查文件是否存在
            index_path = os.path.join(self.embeddings_dir, "index.faiss")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"向量数据库文件不存在: {index_path}")
            
            # 尝试不同的加载方法
            try:
                # 方法1: 标准加载
                vector_store = FAISS.load_local(
                    self.embeddings_dir, 
                    embeddings
                )
            except TypeError as e:
                if "missing 1 required positional argument" in str(e):
                    # 方法2: 兼容旧版本
                    logger.warning("使用兼容模式加载向量数据库")
                    vector_store = FAISS.load_local(
                        self.embeddings_dir, 
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                else:
                    raise
            except Exception as e:
                if "unexpected keyword argument" in str(e):
                    # 方法3: 移除不兼容参数
                    logger.warning("移除不兼容参数重新加载")
                    vector_store = FAISS.load_local(
                        self.embeddings_dir, 
                        embeddings
                    )
                else:
                    raise
            
            logger.info(f"成功加载向量数据库，包含 {vector_store.index.ntotal} 个向量")
            return vector_store
            
        except Exception as e:
            logger.error(f"加载向量数据库失败: {e}")
            raise
    
    def retrieve_relevant_documents(self, query: str, vector_store: FAISS, k: int = 10) -> List[Document]:
        """检索相关文档"""
        try:
            logger.info(f"检索与查询相关的文档: {query[:100]}...")
            
            # 相似性搜索
            docs = vector_store.similarity_search(query, k=k)
            
            logger.info(f"检索到 {len(docs)} 个相关文档片段")
            return docs
            
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            return []
    
    def generate_literature_review(self, 
                                 topic: str, 
                                 documents: List[Document], 
                                 section_type: str = "CURRENT RESEARCH") -> str:
        """使用Ollama生成文献综述"""
        
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(documents[:10]):  # 限制上下文长度
            title = doc.metadata.get('title', f'Document {i+1}')
            content = doc.page_content[:1000]  # 限制每个文档片段长度
            context_parts.append(f"[{title}]\n{content}\n")
        
        context = "\n".join(context_parts)
        
        # 构建系统提示
        system_prompt = f"""你是一个专业的化学化工领域学术文献综述专家。请基于提供的文献内容，生成高质量的学术综述。

要求：
1. 保持学术严谨性和客观性
2. 正确引用文献（使用提供的标题）
3. 保持逻辑清晰的结构
4. 重点关注化学化工相关的技术细节
5. 如果涉及化学公式，请保持原有的LaTeX格式
6. 字数控制在1500-2500字

综述类型：{section_type}"""

        # 构建用户提示
        user_prompt = f"""请基于以下文献内容，围绕主题"{topic}"生成一篇{section_type}类型的学术综述：

=== 文献内容 ===
{context}

=== 要求 ===
- 主题：{topic}
- 类型：{section_type}
- 重点关注化学化工领域的研究进展和技术发展
- 确保内容准确、逻辑清晰、引用规范

请开始生成综述："""

        try:
            logger.info(f"开始生成{section_type}综述，主题: {topic}")
            
            # 使用Ollama生成文本
            response = generate_text(
                prompt=user_prompt,
                model=self.main_model,
                system=system_prompt,
                task_type="literature_generation",  # 使用文献生成的特殊配置
                temperature=0.3,
                max_tokens=8192
            )
            
            logger.info(f"成功生成综述，长度: {len(response)} 字符")
            return response
            
        except Exception as e:
            logger.error(f"生成综述失败: {e}")
            return f"生成综述时出现错误: {str(e)}"
    
    def process_chapter(self, chapter_file: str) -> Dict[str, str]:
        """处理单个章节文件"""
        
        # 读取章节内容
        try:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                chapter_content = f.read()
        except Exception as e:
            logger.error(f"读取章节文件失败: {e}")
            return {}
        
        # 提取章节主题
        chapter_name = os.path.basename(chapter_file).replace('.md', '')
        
        # 加载向量数据库
        vector_store = self.load_vector_store()
        
        # 定义要生成的综述类型
        section_types = [
            "BACKGROUND KNOWLEDGE",
            "CURRENT RESEARCH", 
            "RESEARCH RECOMMENDATIONS"
        ]
        
        results = {}
        
        for section_type in section_types:
            logger.info(f"处理 {chapter_name} - {section_type}")
            
            # 构建查询
            query = f"{chapter_name} {section_type.lower()}"
            
            # 检索相关文档
            relevant_docs = self.retrieve_relevant_documents(query, vector_store, k=15)
            
            if not relevant_docs:
                logger.warning(f"未找到相关文档: {query}")
                results[section_type] = f"未找到与'{query}'相关的文献内容。"
                continue
            
            # 生成综述
            review = self.generate_literature_review(
                topic=chapter_name,
                documents=relevant_docs,
                section_type=section_type
            )
            
            results[section_type] = review
        
        return results
    
    def save_results(self, results: Dict[str, str], chapter_name: str):
        """保存生成结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for section_type, content in results.items():
            filename = f"{chapter_name}_{section_type.replace(' ', '_')}_consolidated_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # 构建输出数据
            output_data = {
                "chapter": chapter_name,
                "section_type": section_type,
                "content": content,
                "timestamp": timestamp,
                "model_used": self.main_model,
                "generation_method": "ollama"
            }
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"保存结果到: {filepath}")
                
            except Exception as e:
                logger.error(f"保存结果失败: {e}")

def main():
    """主函数"""
    logger.info("=== 开始Ollama文献处理流程 ===")
    
    # 初始化处理器
    try:
        processor = OllamaLiteratureProcessor()
    except Exception as e:
        logger.error(f"初始化处理器失败: {e}")
        return
    
    # 查找章节文件
    chapter_dir = "./chapter_markdowns"
    if not os.path.exists(chapter_dir):
        logger.error(f"章节目录不存在: {chapter_dir}")
        return
    
    chapter_files = [f for f in os.listdir(chapter_dir) if f.endswith('.md')]
    
    if not chapter_files:
        logger.error(f"在 {chapter_dir} 中未找到.md文件")
        return
    
    logger.info(f"找到 {len(chapter_files)} 个章节文件")
    
    # 处理每个章节
    for chapter_file in sorted(chapter_files):
        chapter_path = os.path.join(chapter_dir, chapter_file)
        chapter_name = chapter_file.replace('.md', '')
        
        logger.info(f"\n开始处理章节: {chapter_name}")
        
        try:
            # 处理章节
            results = processor.process_chapter(chapter_path)
            
            if results:
                # 保存结果
                processor.save_results(results, chapter_name)
                logger.info(f"章节 {chapter_name} 处理完成")
            else:
                logger.warning(f"章节 {chapter_name} 未生成任何内容")
                
        except Exception as e:
            logger.error(f"处理章节 {chapter_name} 时出错: {e}")
            continue
    
    logger.info("=== Ollama文献处理流程完成 ===")

if __name__ == "__main__":
    main()