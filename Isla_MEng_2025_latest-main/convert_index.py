#!/usr/bin/env python3
import os
import numpy as np
import faiss
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document

def convert_index():
    print("开始转换索引...")
    
    # 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 加载原始索引
    index_path = "./embeddings/faiss.index"
    metadata_path = "./embeddings/metadata.json"  # 使用JSON格式，更容易处理
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print(f"错误：文件不存在 - {index_path} 或 {metadata_path}")
        return False
    
    # 加载索引和元数据
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 将文档转换为LangChain文档格式
    documents = []
    for meta in metadata:
        doc = Document(
            page_content=meta.get('excerpt', ''),
            metadata={
                'title': meta.get('title', ''),
                'authors': meta.get('authors', []),
                'year': meta.get('year', ''),
                'abstract': meta.get('abstract', ''),
                'file_name': meta.get('file_name', ''),
                'path': meta.get('path', ''),
                'chunk_id': meta.get('chunk_id', 0)
            }
        )
        documents.append(doc)
    
    # 创建Langchain FAISS实例
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # 保存为LangChain格式
    vector_store.save_local("./embeddings")
    print(f"成功转换索引！LangChain兼容的FAISS索引已保存到 ./embeddings")
    return True

if __name__ == "__main__":
    convert_index()