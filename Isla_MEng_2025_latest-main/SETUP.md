
### 1. 检查模型是否可用
```bash
# 检查Ollama服务器连接
curl http://192.168.100.140:11434/api/tags

# 验证模型是否存在
python -c "
from ollama_client import ollama_client
models = ollama_client.list_models()
model_names = [m['name'] for m in models]
target_model = 'qwen3:30b-a3b-instruct-2507-q4_K_M'
print(f'目标模型可用: {target_model in model_names}')
print(f'可用模型: {model_names}')
"
```

### 4. 完整流程测试
```bash
# 步骤1: 生成向量数据库（如果还没有）
python step1.py

# 步骤2: Ollama文献生成
python step2_ollama.py
python convert_index.py |

# 步骤3: AG2多Agent协作改进
python step3_ag2.py

# 步骤4: 可视化（使用原有的step4.py）
python step4_ollama.py

# 步骤5： 导出报告
python export_reports.py
```

## 环境变量配置

如果需要修改配置，可以设置以下环境变量：

```bash
# Ollama服务器配置
export OLLAMA_HOST="192.168.100.140"
export OLLAMA_PORT="11434"

# 模型配置
export MAIN_MODEL="qwen3:30b-a3b-instruct-2507-q4_K_M"
export SMALL_MODEL="qwen3:30b-a3b-instruct-2507-q4_K_M"

# 系统配置
export ENVIRONMENT="test"
export DEBUG="true"
```

