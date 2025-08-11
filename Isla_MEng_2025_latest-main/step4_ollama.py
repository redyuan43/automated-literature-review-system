#!/usr/bin/env python3
"""
Step 4 Ollama版 - 概念图生成模块
基于Ollama模型为文档生成Mermaid概念图
"""

import os
import re
import time
import logging
import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from config import OLLAMA_CONFIG, MODEL_CONFIG, SYSTEM_CONFIG
from ollama_client import generate_text
from chemical_formula_processor import ChemicalFormulaProcessor

def setup_logging(log_level=logging.INFO):
    """Configure logging with both file and console output."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"concept_diagrams_ollama_{timestamp}.log"
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

logger = setup_logging()

class OllamaDiagramGenerator:
    """基于Ollama的概念图生成器"""
    
    def __init__(self):
        self.model = MODEL_CONFIG.main_model
        self.formula_processor = ChemicalFormulaProcessor()
        logger.info(f"初始化概念图生成器，使用模型: {self.model}")
    
    def read_document_file(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """读取文档文件并返回标题和内容"""
        try:
            if file_path.endswith('.json'):
                return self._read_json_file(file_path)
            elif file_path.endswith('.md'):
                return self._read_markdown_file(file_path)
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                return None, None
                
        except Exception as e:
            logger.error(f"读取文件失败 {file_path}: {e}")
            return None, None
    
    def _read_json_file(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """读取JSON文件（step2和step3的输出）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                # step2/step3输出格式
                title = data.get('chapter', '') + ' - ' + data.get('section_type', '')
                content = data.get('content', '')
                
                if len(content) < 50:
                    logger.warning(f"JSON文件内容过短: {file_path}")
                    return title, None
                
                return title, content
            else:
                logger.warning(f"JSON格式不正确: {file_path}")
                return None, None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误 {file_path}: {e}")
            return None, None
    
    def _read_markdown_file(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """读取Markdown文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 从第一行提取标题
            lines = content.split('\n')
            title = None
            for line in lines:
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            if not title:
                title = Path(file_path).stem.replace('_', ' ').title()
            
            content = content.strip()
            
            if len(content) < 50:
                logger.warning(f"Markdown文件内容过短: {file_path}")
                return title, None
            
            return title, content
            
        except Exception as e:
            logger.error(f"读取Markdown文件失败 {file_path}: {e}")
            return None, None
    
    def generate_concept_diagram(self, title: str, content: str) -> Optional[str]:
        """使用Ollama生成概念图"""
        try:
            logger.info(f"为文档生成概念图: {title}")
            
            # 保护化学公式
            protected_content = self.formula_processor.preserve_chemical_content(content)
            
            # 构建系统提示
            system_prompt = """你是一个专业的知识图谱和可视化专家。你的任务是为学术文档创建清晰、有意义的Mermaid概念图。

要求：
1. 分析文档内容，提取关键概念和它们之间的关系
2. 创建一个结构清晰的Mermaid图表
3. 使用适当的Mermaid图表类型（flowchart、mindmap、graph等）
4. 包含5-12个核心概念节点
5. 确保关系连接有意义且准确
6. 使用中英文混合标签以符合学术习惯
7. 只返回Mermaid代码块，不要其他解释文字

⚠️ 重要语法规则：
- 节点标签不能包含: $、()、[]、{}、特殊符号
- 数学公式用文字描述，如: "平方根公式" 而不是 "$\\sqrt{x}$"
- 上下标用普通文字，如: "x0" 而不是 "x₀"
- 希腊字母用英文，如: "alpha" 而不是 "α"
- 避免复杂的数学表达式

图表类型选择指南：
- 概念关系: 使用 graph TD 或 flowchart TD
- 层次结构: 使用 mindmap
- 流程过程: 使用 flowchart LR
- 分类体系: 使用 graph TB"""

            # 构建用户提示
            user_prompt = f"""请为以下学术文档创建一个Mermaid概念图：

文档标题：{title}

文档内容：
{protected_content[:8000]}  # 使用前8K字符，保持足够信息

请基于文档内容创建概念图，突出主要概念及其关系。只返回Mermaid代码块。"""

            # 调用Ollama生成
            response = generate_text(
                prompt=user_prompt,
                model=self.model,
                system=system_prompt,
                task_type="diagram_generation",
                temperature=0.3,  # 较低温度确保结构化输出
                max_tokens=4096
            )
            
            # 提取Mermaid代码
            mermaid_code = self._extract_mermaid_code(response)
            
            if mermaid_code:
                # 清理Mermaid代码中的问题字符
                mermaid_code = self._clean_mermaid_code(mermaid_code)
                # 恢复化学公式
                mermaid_code = self.formula_processor.restore_chemical_content(mermaid_code)
                logger.info("成功生成概念图")
                return f"```mermaid\n{mermaid_code}\n```"
            else:
                logger.warning("未能从响应中提取有效的Mermaid代码")
                return None
                
        except Exception as e:
            logger.error(f"生成概念图时出错: {e}")
            return None
    
    def _extract_mermaid_code(self, response: str) -> Optional[str]:
        """从响应中提取Mermaid代码"""
        
        # 方法1: 提取 ```mermaid ... ``` 块
        if "```mermaid" in response:
            try:
                parts = response.split("```mermaid", 1)[1].split("```", 1)
                if parts:
                    code = parts[0].strip()
                    if self._validate_mermaid_code(code):
                        return code
            except Exception:
                pass
        
        # 方法2: 提取任何包含Mermaid关键词的代码块
        if "```" in response:
            try:
                blocks = response.split("```")
                for i in range(1, len(blocks), 2):
                    block = blocks[i].strip()
                    if block.startswith("mermaid"):
                        code = block[7:].strip()
                    else:
                        code = block
                    
                    if self._validate_mermaid_code(code):
                        return code
            except Exception:
                pass
        
        # 方法3: 直接查找Mermaid语法
        mermaid_keywords = ["graph ", "flowchart ", "mindmap", "classDiagram", "sequenceDiagram"]
        for keyword in mermaid_keywords:
            if keyword in response:
                try:
                    start_idx = response.find(keyword)
                    # 查找可能的结束位置
                    end_markers = ["\n```", "\n\n#", "\n\n**", "\n\n总结"]
                    end_idx = len(response)
                    
                    for marker in end_markers:
                        marker_idx = response.find(marker, start_idx)
                        if marker_idx != -1 and marker_idx < end_idx:
                            end_idx = marker_idx
                    
                    code = response[start_idx:end_idx].strip()
                    if self._validate_mermaid_code(code):
                        return code
                except Exception:
                    pass
        
        return None
    
    def _clean_mermaid_code(self, code: str) -> str:
        """清理Mermaid代码中的问题字符和语法"""
        
        # 记录原始代码长度
        original_length = len(code)
        
        # 1. 清理节点标签中的数学公式符号
        # 替换LaTeX数学符号
        code = re.sub(r'\$[^$]*\$', lambda m: self._math_to_text(m.group(0)), code)
        
        # 2. 替换特殊字符和数学符号
        replacements = {
            # 下标
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', '₅': '5',
            '₆': '6', '₇': '7', '₈': '8', '₉': '9',
            # 上标
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
            # 希腊字母小写
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'θ': 'theta', 'λ': 'lambda', 'μ': 'mu',
            'ν': 'nu', 'π': 'pi', 'ρ': 'rho', 'σ': 'sigma',
            'τ': 'tau', 'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
            # 希腊字母大写
            'Α': 'Alpha', 'Β': 'Beta', 'Γ': 'Gamma', 'Δ': 'Delta',
            'Ε': 'Epsilon', 'Θ': 'Theta', 'Λ': 'Lambda', 'Μ': 'Mu',
            'Ν': 'Nu', 'Π': 'Pi', 'Ρ': 'Rho', 'Σ': 'Sigma',
            'Τ': 'Tau', 'Φ': 'Phi', 'Χ': 'Chi', 'Ψ': 'Psi', 'Ω': 'Omega',
            # 数学符号
            '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=',
            '∞': 'inf', '∫': 'integral', '∑': 'sum', '∏': 'product',
            '∂': 'partial', '∇': 'nabla', '∈': 'in', '∉': 'not_in',
            '⁻¹': 'inverse', '⁻²': 'negative2', '⁻³': 'negative3',
            # 其他特殊字符
            '×': 'x', '÷': '/', '±': '+/-', '∓': '-/+',
            '√': 'sqrt', '∛': 'cbrt', '∜': '4thrt'
        }
        
        for old, new in replacements.items():
            code = code.replace(old, new)
        
        # 3. 清理节点标签中的问题字符（在[]内的内容）
        def clean_node_content(match):
            content = match.group(1)
            
            # 移除或替换导致Mermaid解析错误的字符
            problem_chars = {
                '(': '', ')': '', '{': '', '}': '',
                '|': ' ', '&': 'and', '<': 'lt', '>': 'gt',
                '"': "'", '`': "'", '^': '', '_': ' ',
                ':': ' ', ';': ' ', '=': ' equals ',
                '[': '', ']': '', '\\': '/', '/': ' ',
                '#': 'num', '%': 'percent', '@': 'at',
                'Σ': 'Sigma', 'σ': 'sigma', 'E': 'E'  # 处理求和符号
            }
            
            for char, replacement in problem_chars.items():
                content = content.replace(char, replacement)
            
            # 清理多余空格
            content = re.sub(r'\s+', ' ', content).strip()
            
            # 处理不平衡的括号和特殊结构
            # 移除连续的特殊字符
            content = re.sub(r'[\]\[\)\(]+', '', content)
            # 处理数学表达式的简化描述
            content = re.sub(r'\b\d+\]', '', content)  # 移除类似 "2]" 的结构
            content = re.sub(r'obs\s*-\s*', 'observed minus ', content)  # 处理 "obs -" 
            content = re.sub(r'\bt\]', 't', content)  # 处理 "t]"
            # 彻底清理所有可能导致解析错误的字符组合
            content = re.sub(r'[.\]]+$', '', content)  # 移除结尾的 "...]]" 等
            content = re.sub(r'\.{3,}', '...', content)  # 标准化省略号
            # 移除任何剩余的 ]) 组合
            content = re.sub(r'\]\)', '', content)
            content = re.sub(r'\)\]', '', content)
            # 移除数字后面的方括号
            content = re.sub(r'\d+\]', '', content)
            
            # 限制长度
            if len(content) > 50:
                content = content[:47] + "..."
            
            # 如果内容为空，提供默认值
            if not content:
                content = "概念"
            
            return f"[{content}]"
        
        code = re.sub(r'\[([^\]]+)\]', clean_node_content, code)
        
        # 3.5. 简单清理每行末尾的多余内容
        # 移除行末的多余方括号和引用信息
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # 如果行包含 --> 和 []，检查是否有行末多余内容
            if '-->' in line and '[' in line and ']' in line:
                # 找到最后一个完整的节点定义结束位置
                last_bracket_pos = line.rfind(']')
                if last_bracket_pos != -1:
                    # 检查方括号后是否还有多余内容
                    after_bracket = line[last_bracket_pos + 1:].strip()
                    if after_bracket:
                        # 移除多余内容，保留到最后一个完整节点
                        line = line[:last_bracket_pos + 1]
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines)
        
        # 4. 清理连接标签中的问题字符
        def clean_connection_label(match):
            full_match = match.group(0)
            label = match.group(1)
            # 简化连接标签
            label = label.replace('(', '').replace(')', '')
            label = label.replace('{', '').replace('}', '')
            if len(label) > 20:
                label = label[:17] + "..."
            return f"|{label}|"
        
        code = re.sub(r'\|([^|]+)\|', clean_connection_label, code)
        
        # 记录清理结果
        logger.debug(f"Mermaid代码清理: {original_length} -> {len(code)} 字符")
        
        return code
    
    def _math_to_text(self, latex_formula: str) -> str:
        """将LaTeX数学公式转换为文本描述"""
        formula = latex_formula.strip('$')
        
        # 常见数学表达式的文本替换
        math_replacements = {
            r'\\frac\{([^}]+)\}\{([^}]+)\}': r'\1 除以 \2',
            r'\\sqrt\{([^}]+)\}': r'\1 的平方根',
            r'\\ln\(([^)]+)\)': r'ln(\1)',
            r'\\log\(([^)]+)\)': r'log(\1)',
            r'\\alpha': 'alpha',
            r'\\beta': 'beta',
            r'\\gamma': 'gamma',
            r'\\pi': 'pi',
            r'\\sigma': 'sigma',
            r'\\mu': 'mu',
            r'\\Phi': 'Phi',
            r'\^': '的',
            r'_': '',
        }
        
        for pattern, replacement in math_replacements.items():
            formula = re.sub(pattern, replacement, formula)
        
        # 如果还是太复杂，直接返回简化描述
        if len(formula) > 30 or any(char in formula for char in ['\\', '{', '}', '^', '_']):
            return "数学公式"
        
        return formula
    
    def _validate_mermaid_code(self, code: str) -> bool:
        """验证Mermaid代码的基本有效性"""
        if not code or len(code) < 10:
            return False
        
        # 检查是否包含基本Mermaid语法
        mermaid_patterns = [
            "graph ", "flowchart ", "mindmap", "classDiagram", 
            "sequenceDiagram", "-->", "---", "==>"
        ]
        
        has_mermaid_syntax = any(pattern in code for pattern in mermaid_patterns)
        if not has_mermaid_syntax:
            return False
        
        # 检查是否是示例代码（排除通用示例）
        generic_examples = [
            "Main Topic", "Topic 1", "Subtopic A", "Example Node",
            "Node A", "Node B", "概念A", "概念B"
        ]
        
        is_generic = sum(1 for example in generic_examples if example in code) >= 2
        if is_generic:
            return False
        
        return True

    def process_file(self, file_path: str, output_dir: str) -> bool:
        """处理单个文件生成概念图"""
        try:
            logger.info(f"处理文件: {file_path}")
            
            title, content = self.read_document_file(file_path)
            if not content:
                logger.warning(f"无法获取有效内容: {file_path}")
                return False
            
            # 生成概念图
            diagram = self.generate_concept_diagram(title, content)
            if not diagram:
                logger.warning(f"无法生成概念图: {file_path}")
                return False
            
            # 保存结果
            base_filename = Path(file_path).stem
            output_filename = f"{base_filename}_concept_diagram.md"
            output_path = os.path.join(output_dir, output_filename)
            
            # 创建完整的输出文档
            output_content = f"""# 概念图：{title}

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**源文件**: {file_path}
**模型**: {self.model}

## 核心概念关系图

{diagram}

## 图表说明

此概念图展示了文档中的核心概念及其相互关系，帮助理解文档的知识结构。

---
*由 Ollama ({self.model}) 自动生成*
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            logger.info(f"概念图已保存: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return False

def main():
    """主函数"""
    try:
        parser = argparse.ArgumentParser(description="使用Ollama生成文档概念图")
        parser.add_argument("--input_dir", default="outputs", 
                          help="输入目录 (默认: outputs)")
        parser.add_argument("--output_dir", default="concept_diagrams", 
                          help="输出目录 (默认: concept_diagrams)")
        parser.add_argument("--file_pattern", default="*.json", 
                          help="文件匹配模式 (默认: *.json)")
        parser.add_argument("--log_level", default="INFO", 
                          choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                          help="日志级别")
        args = parser.parse_args()
        
        # 设置日志级别
        if args.log_level:
            logger.setLevel(getattr(logging, args.log_level))
        
        logger.info("=== 开始Ollama概念图生成流程 ===")
        logger.info(f"输入目录: {args.input_dir}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"文件模式: {args.file_pattern}")
        logger.info(f"使用模型: {MODEL_CONFIG.main_model}")
        
        # 检查输入目录
        if not os.path.exists(args.input_dir):
            logger.error(f"输入目录不存在: {args.input_dir}")
            return
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 初始化生成器
        generator = OllamaDiagramGenerator()
        
        # 查找待处理文件
        input_path = Path(args.input_dir)
        files_to_process = list(input_path.glob(args.file_pattern))
        
        if not files_to_process:
            logger.warning(f"在 {args.input_dir} 中未找到匹配 {args.file_pattern} 的文件")
            return
        
        logger.info(f"找到 {len(files_to_process)} 个待处理文件")
        
        # 处理文件
        successful = 0
        failed = 0
        
        for file_path in files_to_process:
            logger.info(f"处理文件 {successful + failed + 1}/{len(files_to_process)}: {file_path.name}")
            
            if generator.process_file(str(file_path), args.output_dir):
                successful += 1
            else:
                failed += 1
        
        # 输出统计信息
        logger.info("=== 概念图生成完成 ===")
        logger.info(f"总文件数: {len(files_to_process)}")
        logger.info(f"成功生成: {successful}")
        logger.info(f"生成失败: {failed}")
        logger.info(f"成功率: {successful/len(files_to_process)*100:.1f}%")
        
        if failed > 0:
            logger.warning(f"有 {failed} 个文件处理失败，请检查日志")
        
    except Exception as e:
        logger.error(f"主程序执行失败: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.critical("程序执行出现致命错误", exc_info=True)
        sys.exit(1)