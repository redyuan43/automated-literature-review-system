# config.py - 系统配置文件

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class OllamaConfig:
    """Ollama配置类"""
    host: str = "192.168.100.140"
    port: int = 11434
    base_url: str = None
    timeout: int = 300  # 5分钟超时
    
    def __post_init__(self):
        if self.base_url is None:
            self.base_url = f"http://{self.host}:{self.port}"

@dataclass
class ModelConfig:
    """模型配置类"""
    # 小模型测试环境
    small_model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M"  # Agent使用的小模型
    main_model: str = "qwen3:30b-a3b"  # 主模型
    test_model: str = "qwen3:30b-a3b"   # 测试用模型
    
    # 针对不同任务的特殊配置 (仅在需要时覆盖Ollama默认值)
    task_configs: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.task_configs is None:
            # 只调整关键参数，其他使用Ollama默认值
            self.task_configs = {
                "literature_generation": {
                    "temperature": 0.3  # 学术内容需要更准确
                },
                "agent_review": {
                    "temperature": 0.5  # 评审需要平衡准确性和创造性
                },
                "chemical_analysis": {
                    "temperature": 0.2  # 化学分析需要最高准确性
                },
                "diagram_generation": {
                    "temperature": 0.1,  # 概念图需要结构化思维
                    "max_tokens": 4096   # 图表代码可能较长
                }
            }

@dataclass
class SystemConfig:
    """系统配置类"""
    # 环境配置
    environment: str = "test"  # test | production
    debug: bool = True
    
    # 文件路径
    input_dir: str = "./files_mmd"
    output_dir: str = "./outputs"
    embeddings_dir: str = "./embeddings"
    logs_dir: str = "./logs"
    
    # 处理参数
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_concurrent_requests: int = 5
    
    # Agent配置
    max_agent_rounds: int = 10
    agent_timeout: int = 180
    
    # 化学公式处理
    preserve_formulas: bool = True
    formula_formats: list = None
    
    def __post_init__(self):
        if self.formula_formats is None:
            self.formula_formats = ["latex", "mathml", "unicode"]
        
        # 根据环境调整配置
        if self.environment == "production":
            self.debug = False
            self.max_concurrent_requests = 10

# 全局配置实例
OLLAMA_CONFIG = OllamaConfig()
MODEL_CONFIG = ModelConfig()
SYSTEM_CONFIG = SystemConfig()

# 环境变量覆盖
def load_config_from_env():
    """从环境变量加载配置"""
    
    # Ollama配置
    if os.getenv("OLLAMA_HOST"):
        OLLAMA_CONFIG.host = os.getenv("OLLAMA_HOST")
    if os.getenv("OLLAMA_PORT"):
        OLLAMA_CONFIG.port = int(os.getenv("OLLAMA_PORT"))
    
    # 模型配置
    if os.getenv("MAIN_MODEL"):
        MODEL_CONFIG.main_model = os.getenv("MAIN_MODEL")
    if os.getenv("SMALL_MODEL"):
        MODEL_CONFIG.small_model = os.getenv("SMALL_MODEL")
    
    # 系统配置
    if os.getenv("ENVIRONMENT"):
        SYSTEM_CONFIG.environment = os.getenv("ENVIRONMENT")
    if os.getenv("DEBUG"):
        SYSTEM_CONFIG.debug = os.getenv("DEBUG").lower() == "true"
    
    # 重新计算依赖配置
    OLLAMA_CONFIG.__post_init__()

# 配置验证
def validate_config():
    """验证配置有效性"""
    import requests
    try:
        response = requests.get(f"{OLLAMA_CONFIG.base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"✅ Ollama连接成功: {OLLAMA_CONFIG.base_url}")
            return True
        else:
            print(f"❌ Ollama连接失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama连接异常: {e}")
        return False

# 初始化配置
load_config_from_env()

if __name__ == "__main__":
    print("🔧 系统配置信息:")
    print(f"Ollama服务器: {OLLAMA_CONFIG.base_url}")
    print(f"主模型: {MODEL_CONFIG.main_model}")
    print(f"小模型: {MODEL_CONFIG.small_model}")
    print(f"环境: {SYSTEM_CONFIG.environment}")
    print(f"调试模式: {SYSTEM_CONFIG.debug}")
    
    # 验证连接
    validate_config()