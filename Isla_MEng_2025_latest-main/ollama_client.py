# ollama_client.py - Ollama客户端封装

import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional, Generator
from config import OLLAMA_CONFIG, MODEL_CONFIG, SYSTEM_CONFIG

logger = logging.getLogger(__name__)

class OllamaClient:
    """Ollama API客户端"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.base_url = OLLAMA_CONFIG.base_url
        self.timeout = OLLAMA_CONFIG.timeout
        self.session = requests.Session()
        
        # 应用自定义配置
        if config:
            for key, value in config.items():
                if hasattr(OLLAMA_CONFIG, key):
                    setattr(OLLAMA_CONFIG, key, value)
    
    def generate(self, 
                model: str,
                prompt: str,
                system: str = None,
                temperature: float = None,
                max_tokens: int = None,
                top_p: float = None,
                stream: bool = False) -> Dict[str, Any]:
        """
        生成文本
        
        Args:
            model: 模型名称
            prompt: 用户输入
            system: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top_p参数
            stream: 是否流式输出
            
        Returns:
            生成的响应
        """
        
        # 构建请求数据
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {}
        }
        
        # 添加系统消息
        if system:
            data["system"] = system
        
        # 添加参数 (仅在明确指定时设置，否则使用Ollama默认值)
        if temperature is not None:
            data["options"]["temperature"] = temperature
            
        if max_tokens is not None:
            data["options"]["num_predict"] = max_tokens
            
        if top_p is not None:
            data["options"]["top_p"] = top_p
        
        try:
            if stream:
                return self._stream_generate(data)
            else:
                return self._single_generate(data)
                
        except Exception as e:
            logger.error(f"Ollama生成失败: {e}")
            raise
    
    def _single_generate(self, data: Dict) -> Dict[str, Any]:
        """单次生成"""
        url = f"{self.base_url}/api/generate"
        
        response = self.session.post(
            url, 
            json=data, 
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return {
            "response": result.get("response", ""),
            "done": result.get("done", False),
            "total_duration": result.get("total_duration", 0),
            "load_duration": result.get("load_duration", 0),
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "eval_count": result.get("eval_count", 0),
        }
    
    def _stream_generate(self, data: Dict) -> Generator[Dict[str, Any], None, None]:
        """流式生成"""
        url = f"{self.base_url}/api/generate"
        
        with self.session.post(
            url, 
            json=data, 
            timeout=self.timeout,
            stream=True
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk
                    except json.JSONDecodeError:
                        continue
    
    def chat(self,
             model: str,
             messages: List[Dict[str, str]],
             temperature: float = None,
             max_tokens: int = None,
             stream: bool = False) -> Dict[str, Any]:
        """
        对话接口
        
        Args:
            model: 模型名称
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式输出
            
        Returns:
            对话响应
        """
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {}
        }
        
        # 添加参数 (仅在明确指定时设置，否则使用Ollama默认值)
        if temperature is not None:
            data["options"]["temperature"] = temperature
            
        if max_tokens is not None:
            data["options"]["num_predict"] = max_tokens
        
        try:
            url = f"{self.base_url}/api/chat"
            
            if stream:
                return self._stream_chat(url, data)
            else:
                response = self.session.post(url, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                
                return {
                    "message": result.get("message", {}),
                    "done": result.get("done", False),
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                }
                
        except Exception as e:
            logger.error(f"Ollama对话失败: {e}")
            raise
    
    def _stream_chat(self, url: str, data: Dict) -> Generator[Dict[str, Any], None, None]:
        """流式对话"""
        with self.session.post(
            url, 
            json=data, 
            timeout=self.timeout,
            stream=True
        ) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        yield chunk
                    except json.JSONDecodeError:
                        continue
    
    def list_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get("models", [])
            
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
    
    def pull_model(self, model: str) -> bool:
        """拉取模型"""
        try:
            url = f"{self.base_url}/api/pull"
            data = {"name": model}
            
            response = self.session.post(url, json=data, timeout=600)  # 10分钟超时
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"拉取模型失败: {e}")
            return False
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            models = self.list_models()
            return len(models) >= 0
        except:
            return False

# 全局客户端实例
ollama_client = OllamaClient()

# 便捷函数
def generate_text(prompt: str, 
                 model: str = None, 
                 system: str = None,
                 task_type: str = None,
                 **kwargs) -> str:
    """
    便捷的文本生成函数
    
    Args:
        prompt: 输入提示
        model: 模型名称，默认使用配置中的主模型
        system: 系统提示
        task_type: 任务类型，用于获取特殊配置
        **kwargs: 其他参数
        
    Returns:
        生成的文本
    """
    
    if model is None:
        model = MODEL_CONFIG.main_model
    
    # 应用任务特定配置
    if task_type and task_type in MODEL_CONFIG.task_configs:
        task_config = MODEL_CONFIG.task_configs[task_type]
        for key, value in task_config.items():
            if key not in kwargs:
                kwargs[key] = value
    
    result = ollama_client.generate(
        model=model,
        prompt=prompt,
        system=system,
        **kwargs
    )
    
    return result["response"]

def chat_with_model(messages: List[Dict[str, str]], 
                   model: str = None,
                   **kwargs) -> str:
    """
    便捷的对话函数
    
    Args:
        messages: 对话历史
        model: 模型名称
        **kwargs: 其他参数
        
    Returns:
        模型回复
    """
    
    if model is None:
        model = MODEL_CONFIG.small_model  # 对话默认使用小模型
    
    result = ollama_client.chat(
        model=model,
        messages=messages,
        **kwargs
    )
    
    return result["message"]["content"]

if __name__ == "__main__":
    # 测试连接
    client = OllamaClient()
    
    print("🔍 检查Ollama连接...")
    if client.health_check():
        print("✅ Ollama连接正常")
        
        # 列出可用模型
        models = client.list_models()
        print(f"📋 可用模型: {[m['name'] for m in models]}")
        
        # 测试生成
        if models:
            test_model = models[0]['name']
            print(f"🧪 测试模型: {test_model}")
            
            response = generate_text(
                prompt="Hello, how are you?",
                model=test_model,
                max_tokens=50
            )
            print(f"💬 测试响应: {response}")
        
    else:
        print("❌ Ollama连接失败")