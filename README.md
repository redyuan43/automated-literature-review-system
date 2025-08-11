# Automated Multi-Agent Academic Literature Review Generation System

## 项目概述

这是一个基于大语言模型的端到端自动化学术文献综述生成系统，采用多智能体协作架构实现从PDF原始文献到结构化报告的完整处理流程。系统包含两个核心模块：PDF文档OCR转换（pdf2md）和文献综述生成（Isla_MEng_2025）。

## 完整处理流程

### 第一阶段：PDF文档处理（pdf2md）

使用DotsOCR进行高精度学术文档OCR，支持数学公式和化学反应式保护。

### 第二阶段：文献综述生成（Isla_MEng_2025）

基于Ollama架构的多Agent协作系统，生成结构化文献综述报告。

## 系统架构

### 核心组件

#### PDF处理模块（pdf2md）
1. **OCR服务**
   - 基于vLLM的OCR服务
   - 支持数学公式和化学反应式识别
   - 多GPU并行处理

2. **公式保护系统**
   - `chemical_formula_processor.py`: 核心公式保护引擎
   - `dots_ocr/utils/formula_protection_utils.py`: OCR集成工具
   - 防止公式在文本分块时被切断

#### 文献处理模块（Isla_MEng_2025）
1. **数据准备与嵌入 (step1.py)**
   - 处理 `.mmd` 格式的科研文献文件
   - 提取元数据（标题、作者、年份、摘要）
   - 使用 SentenceTransformer 生成文本嵌入
   - 用 FAISS 建立向量检索库

2. **Ollama文献生成 (step2_ollama.py)**
   - 使用远程Ollama服务器（192.168.100.140）
   - 加载qwen3:30b模型进行推理
   - 实现与向量库的连接

3. **多Agent协作改进 (step3_ag2.py)**
   - 基于AG2框架的多智能体评审
   - 包括技术性、结构、可读性、引用准确性等自动分析
   - 通过 moderator agent 综合各方建议

4. **可视化与输出 (step4_ollama.py)**
   - 为每篇文档生成知识结构图（Mermaid图）
   - 结构化报告导出

5. **报告导出 (export_reports.py)**
   - 生成最终的文献综述报告
   - 支持多种格式输出

## 技术栈

### PDF处理
- **OCR**: DotsOCR (基于vLLM)
- **公式保护**: 数学公式和化学反应式识别
- **并行处理**: CUDA多GPU支持

### 文献分析
- **LLM**: Ollama (qwen3:30b), Gemma-3-27B, Mistral-7B
- **嵌入**: SentenceTransformer (all-MiniLM-L6-v2)
- **向量库**: FAISS
- **Agent框架**: AG2 (原AutoGen)
- **效率优化**: BitsAndBytes (4-bit量化), PEFT/LoRA
- **评估**: spaCy, TextStat, Scikit-learn

## 部署要求

### 硬件要求
- Linux系统，英伟达CUDA支持
- 至少2张GPU（推荐3090 24G）
- 最少28GB RAM
- Python 3.10.17

### 服务依赖
- Ollama服务器（<localhost>:11434）- 根据客户部署的ollama来设置
- vLLM服务（本地部署）

## 快速开始

### 阶段1：PDF转换为MMD格式

1. **启动DotsOCR服务**
```bash
# 激活OCR环境
conda activate dots_ocr

# 配置模型路径
export hf_model_path=./weights/DotsOCR
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH

# 启动vLLM服务器
CUDA_VISIBLE_DEVICES=0,1 vllm serve ${hf_model_path} \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --chat-template-content-format string \
  --served-model-name DotsOCR \
  --trust-remote-code
```

2. **运行PDF处理**
```bash
# 处理单个PDF文件
python3 dots_ocr/parser.py pdf/your_paper.pdf --num_thread 64 --model_name DotsOCR

# 批量处理（带公式保护）
python3 reprocess_ocr_with_protection.py
```

### 阶段2：文献综述生成

1. **环境设置**
```bash
# 切换到主项目目录
cd Isla_MEng_2025_latest-main/

# 激活主环境
conda activate new_env

# 配置Ollama连接
export OLLAMA_HOST=<localhost>
export MAIN_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M
```

2. **验证Ollama服务**
```bash
# 检查服务器连接
curl http://<localhost>:11434/api/tags

# 验证模型可用性
python -c "
from ollama_client import ollama_client
models = ollama_client.list_models()
print('目标模型可用:', 'qwen3:30b-a3b-instruct-2507-q4_K_M' in [m['name'] for m in models])
"
```

3. **执行完整流程**
```bash
# 步骤1: 生成向量数据库
python step1.py

# 步骤2: Ollama文献生成
python step2_ollama.py
python convert_index.py

# 步骤3: AG2多Agent协作改进
python step3_ag2.py

# 步骤4: 可视化
python step4_ollama.py

# 步骤5: 导出报告
python export_reports.py
```

## 项目结构

```
├── pdf2md/                       # PDF处理模块
│   ├── dots_ocr/                # DotsOCR核心
│   │   ├── parser.py           # PDF解析器
│   │   └── utils/              # 工具函数
│   ├── chemical_formula_processor.py  # 公式保护
│   └── reprocess_ocr_with_protection.py  # 批处理工具
│
├── Isla_MEng_2025_latest-main/   # 文献综述生成模块
│   ├── step1.py                # 文档索引和嵌入
│   ├── step2_ollama.py         # Ollama文献生成
│   ├── step3_ag2.py            # AG2多Agent协作
│   ├── step4_ollama.py         # 可视化输出
│   ├── export_reports.py       # 报告导出
│   ├── convert_index.py        # 索引转换
│   ├── ollama_client.py        # Ollama客户端
│   ├── test_new_architecture.py # 系统验证
│   ├── files_mmd/              # 输入文档目录
│   ├── embeddings/             # 向量存储
│   ├── outputs/                # 输出报告
│   └── logs/                   # 运行日志
│
├── environment.yml              # Conda环境配置
└── README.md                   # 项目说明
```

## 输入输出说明

### 输入格式
- **PDF文档**: 原始学术论文（pdf/目录）
- **MMD文档**: OCR转换后的Markdown格式（files_mmd/目录）

### 输出内容
- **向量数据库**: FAISS索引文件（embeddings/目录）
- **文献综述报告**: 结构化分析报告（outputs/目录）
- **知识图谱**: Mermaid格式的概念关系图
- **评估报告**: 多维度质量评分

## 评估体系

系统包含完整的质量评估框架：
- **技术深度 (45%)**: 技术术语密度、概念层次深度、句法复杂度
- **清晰度 (35%)**: Flesch可读性、术语定义、示例数量
- **结构 (20%)**: 主题建模、局部连贯性、主题一致性

## 故障排除

### DotsOCR服务问题
```bash
# 检查vLLM服务状态
curl http://localhost:8000/v1/models

# 重启服务
pkill -f vllm
# 重新启动vLLM服务器
```

### Ollama连接问题
```bash
# 测试网络连接
ping <localhost>

# 检查Ollama服务
curl http://<localhost>:11434/api/tags

# 设置环境变量
export OLLAMA_HOST=<localhost>
export OLLAMA_PORT="11434"
```

### GPU内存问题
- 减少批处理大小
- 调整gpu-memory-utilization参数
- 使用单GPU模式

## 许可证

MIT License

## 联系方式

- GitHub: redyuan43@gmail.com 