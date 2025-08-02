# Automated Multi-Agent Academic Literature Review Generation System

## 项目概述

这是一个基于大语言模型的端到端自动化学术文献综述生成系统，采用多智能体协作架构实现从原始文献到结构化报告的完整处理流程。

## 系统架构

### 核心组件

1. **数据准备与嵌入 (step1.py)**
   - 处理 `.mmd` 格式的科研文献文件
   - 提取元数据（标题、作者、年份、摘要）
   - 使用 SentenceTransformer 生成文本嵌入
   - 用 FAISS 建立向量检索库

2. **本地LLM推理与向量库加载 (step2.py)**
   - 加载本地大语言模型（Gemma-3-27B）
   - 实现与向量库的连接
   - 使用 LangChain 的 FAISS 向量库封装
   - 自定义 LLM client (CustomGemmaClient) 统一推理接口

3. **多Agent自动评审与改写建议生成 (step3.py)**
   - 实现"多智能体"自动化评审流程
   - 包括技术性、结构、可读性、引用准确性等自动分析
   - 通过 moderator agent 综合各方建议，给出最终改写指令

4. **自动改写、结构化输出与可视化**
   - `rewrite_function.py`: 封装LLM调用细节与多轮对话处理
   - `step4.py`: 为每篇文档生成知识结构图（Mermaid图）

5. **自动评分评价 (final_evaluation_openrouter.py)**
   - 对生成内容进行多维度评分
   - 技术深度、语言复杂度、结构、引用准确性等

6. **微调 (fine_tune.py)**
   - 在自有数据集上进行LLM微调
   - 支持从 markdown/json 结构提取文本
   - 结合 LoRA、BitsAndBytes 做高效微调

## 技术栈

- **LLM**: Gemma-3-27B, Mistral-7B
- **嵌入**: SentenceTransformer (all-MiniLM-L6-v2)
- **向量库**: FAISS
- **Agent框架**: AutoGen
- **效率优化**: BitsAndBytes (4-bit量化), PEFT/LoRA
- **评估**: spaCy, TextStat, Scikit-learn

## 部署要求

- 超算集群，Linux系统，英伟达CUDA集群
- 支持SLURM批处理系统
- 至少2张A100 80G GPU
- Python 3.10.17

## 快速开始

1. **克隆仓库**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **创建环境**
```bash
conda env create -f environment.yml
conda activate new_env
```

3. **配置环境变量**
```bash
export HUGGINGFACE_TOKEN=your_token_here
export OPENROUTER_API_KEY=your_api_key_here
```

4. **运行系统**
```bash
sbatch batch_script.sh
```

## 项目结构

```
├── step1.py                     # 文档索引和嵌入
├── step2.py                     # 初始报告生成
├── step3.py                     # 多Agent评审和改进
├── step4.py                     # 概念图生成
├── rewrite_function.py          # 文本重写工具
├── fine_tune.py                # 模型微调
├── final_evaluation_openrouter.py # 最终评估
├── batch_script.sh             # SLURM批处理脚本
├── environment.yml             # 环境配置
└── README.md                   # 项目说明
```

## 评估体系

系统包含完整的质量评估框架：
- **技术深度 (45%)**: 技术术语密度、概念层次深度、句法复杂度
- **清晰度 (35%)**: Flesch可读性、术语定义、示例数量
- **结构 (20%)**: 主题建模、局部连贯性、主题一致性

## 许可证

MIT License

## 联系方式

- GitHub: redyuan43@gmail.com 