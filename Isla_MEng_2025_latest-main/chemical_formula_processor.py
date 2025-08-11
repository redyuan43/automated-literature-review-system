# chemical_formula_processor.py - 化学公式处理模块

import re
import logging
from typing import List, Tuple, Dict, Any
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ChemicalFormulaProcessor:
    """化学公式处理器"""
    
    def __init__(self):
        # LaTeX化学公式模式
        self.latex_patterns = [
            r'\$\$.*?\$\$',  # 独立公式 $$...$$
            r'\$.*?\$',      # 行内公式 $...$
            r'\\begin\{equation\}.*?\\end\{equation\}',  # equation环境
            r'\\begin\{align\}.*?\\end\{align\}',        # align环境
            r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',  # eqnarray环境
            r'\\begin\{chemical\}.*?\\end\{chemical\}',   # chemical环境
            r'\\ce\{.*?\}',   # mhchem包的\\ce命令
            r'\\cee\{.*?\}',  # mhchem包的\\cee命令
        ]
        
        # 化学符号模式
        self.chemical_patterns = [
            r'[A-Z][a-z]?(?:\d+)?(?:[+-])?',  # 基本化学元素和离子
            r'[A-Z][a-z]?\d*(?:\([IVX]+\))?', # 包含价态的元素
            r'(?:[A-Z][a-z]?\d*)+',           # 化合物分子式
            r'\d*[A-Z][a-z]?\d*(?:[+-]\d*)?', # 更复杂的化学式
        ]
        
        # 常见化学反应箭头和符号
        self.reaction_symbols = [
            r'→', r'←', r'↔', r'⇌', r'⇄',  # 反应箭头
            r'-->', r'<--', r'<-->', r'<=>',  # ASCII箭头
            r'Δ', r'∆',  # 加热符号
            r'⇌', r'⟷',  # 平衡箭头
            r'↑', r'↓',  # 气体和沉淀符号
            r'°C', r'°F', r'K',  # 温度单位
            r'atm', r'Pa', r'bar', r'torr',  # 压力单位
            r'mol', r'mmol', r'μmol',  # 摩尔单位
            r'M', r'mM', r'μM',  # 浓度单位
        ]
        
        # 编译正则表达式
        self.compiled_latex_patterns = [re.compile(pattern, re.DOTALL) for pattern in self.latex_patterns]
        self.compiled_chemical_patterns = [re.compile(pattern) for pattern in self.chemical_patterns]
        self.compiled_reaction_symbols = [re.compile(pattern) for pattern in self.reaction_symbols]
    
    def extract_latex_formulas(self, text: str) -> List[Tuple[str, int, int]]:
        """
        提取LaTeX格式的化学公式
        
        Returns:
            List[Tuple[str, int, int]]: [(公式内容, 开始位置, 结束位置), ...]
        """
        formulas = []
        
        for pattern in self.compiled_latex_patterns:
            for match in pattern.finditer(text):
                formula = match.group(0)
                start, end = match.span()
                formulas.append((formula, start, end))
        
        # 按位置排序，避免重叠
        formulas.sort(key=lambda x: x[1])
        
        # 去除重叠的公式（保留最长的）
        filtered_formulas = []
        for formula, start, end in formulas:
            overlap = False
            for existing_formula, existing_start, existing_end in filtered_formulas:
                if not (end <= existing_start or start >= existing_end):
                    overlap = True
                    break
            if not overlap:
                filtered_formulas.append((formula, start, end))
        
        return filtered_formulas
    
    def detect_chemical_entities(self, text: str) -> List[Tuple[str, int, int]]:
        """
        检测化学实体（分子式、离子等）
        
        Returns:
            List[Tuple[str, int, int]]: [(化学实体, 开始位置, 结束位置), ...]
        """
        entities = []
        
        for pattern in self.compiled_chemical_patterns:
            for match in pattern.finditer(text):
                entity = match.group(0)
                start, end = match.span()
                
                # 过滤掉过短或明显不是化学式的匹配
                if len(entity) >= 2 and self._is_likely_chemical(entity):
                    entities.append((entity, start, end))
        
        return entities
    
    def _is_likely_chemical(self, entity: str) -> bool:
        """判断是否可能是化学实体"""
        # 基本启发式规则
        if len(entity) < 2:
            return False
        
        # 必须以大写字母开头
        if not entity[0].isupper():
            return False
        
        # 包含数字的可能性更高
        has_number = any(c.isdigit() for c in entity)
        
        # 包含小写字母的可能性更高（元素符号）
        has_lowercase = any(c.islower() for c in entity)
        
        # 排除常见的英文单词
        common_words = {'The', 'This', 'That', 'For', 'And', 'But', 'Or'}
        if entity in common_words:
            return False
        
        return has_lowercase or has_number
    
    def detect_reaction_symbols(self, text: str) -> List[Tuple[str, int, int]]:
        """检测化学反应符号"""
        symbols = []
        
        for pattern in self.compiled_reaction_symbols:
            for match in pattern.finditer(text):
                symbol = match.group(0)
                start, end = match.span()
                symbols.append((symbol, start, end))
        
        return symbols
    
    def preserve_chemical_content(self, text: str) -> str:
        """
        保留文本中的化学内容，确保公式格式不丢失
        
        对于已经是LaTeX格式的MMD文件，直接返回原文本，
        因为公式已经被正确保护了。
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本（对于MMD文件，返回原文本）
        """
        
        # 检查是否包含LaTeX公式标记，如果有则认为已经被正确保护
        latex_block_pattern = r'\$\$.*?\$\$'
        latex_inline_pattern = r'\$[^$]+\$'
        
        if re.search(latex_block_pattern, text, re.DOTALL) or re.search(latex_inline_pattern, text):
            # 文本已包含LaTeX公式，直接返回不做额外保护
            latex_formulas = self.extract_latex_formulas(text)
            logger.info(f"检测到 {len(latex_formulas)} 个LaTeX公式，保持原有格式")
            return text
        
        # 如果没有LaTeX公式，则进行传统的化学内容保护
        # 1. 提取LaTeX公式
        latex_formulas = self.extract_latex_formulas(text)
        
        # 2. 提取化学实体
        chemical_entities = self.detect_chemical_entities(text)
        
        # 3. 提取反应符号
        reaction_symbols = self.detect_reaction_symbols(text)
        
        # 记录发现的内容
        if latex_formulas:
            logger.info(f"发现 {len(latex_formulas)} 个LaTeX公式")
        if chemical_entities:
            logger.info(f"发现 {len(chemical_entities)} 个化学实体")
        if reaction_symbols:
            logger.info(f"发现 {len(reaction_symbols)} 个反应符号")
        
        # 4. 标记化学内容（为后续处理做准备）
        processed_text = self._mark_chemical_content(text, latex_formulas, chemical_entities, reaction_symbols)
        
        return processed_text
    
    def _mark_chemical_content(self, text: str, 
                              latex_formulas: List[Tuple[str, int, int]],
                              chemical_entities: List[Tuple[str, int, int]],
                              reaction_symbols: List[Tuple[str, int, int]]) -> str:
        """标记化学内容"""
        
        # 合并所有需要保护的内容
        all_items = []
        
        # LaTeX公式（优先级最高）
        for formula, start, end in latex_formulas:
            all_items.append((start, end, f"<CHEM_LATEX>{formula}</CHEM_LATEX>", "latex"))
        
        # 化学实体
        for entity, start, end in chemical_entities:
            all_items.append((start, end, f"<CHEM_ENTITY>{entity}</CHEM_ENTITY>", "entity"))
        
        # 反应符号
        for symbol, start, end in reaction_symbols:
            all_items.append((start, end, f"<CHEM_SYMBOL>{symbol}</CHEM_SYMBOL>", "symbol"))
        
        # 按位置排序
        all_items.sort(key=lambda x: x[0])
        
        # 去除重叠（LaTeX公式优先级最高）
        filtered_items = []
        for start, end, replacement, item_type in all_items:
            overlap = False
            for existing_start, existing_end, _, existing_type in filtered_items:
                if not (end <= existing_start or start >= existing_end):
                    # 如果有重叠，LaTeX公式优先
                    if existing_type == "latex" or (existing_type != "latex" and item_type == "latex"):
                        overlap = True
                    break
            if not overlap:
                filtered_items.append((start, end, replacement, item_type))
        
        # 应用替换
        result = ""
        last_end = 0
        
        for start, end, replacement, _ in filtered_items:
            # 添加前面的文本
            result += text[last_end:start]
            # 添加标记的化学内容
            result += replacement
            last_end = end
        
        # 添加剩余文本
        result += text[last_end:]
        
        return result
    
    def restore_chemical_content(self, text: str) -> str:
        """恢复化学内容的原始格式"""
        
        # 恢复LaTeX公式
        text = re.sub(r'<CHEM_LATEX>(.*?)</CHEM_LATEX>', r'\1', text)
        
        # 恢复化学实体
        text = re.sub(r'<CHEM_ENTITY>(.*?)</CHEM_ENTITY>', r'\1', text)
        
        # 恢复反应符号
        text = re.sub(r'<CHEM_SYMBOL>(.*?)</CHEM_SYMBOL>', r'\1', text)
        
        return text

def test_chemical_formula_processor():
    """测试化学公式处理器"""
    
    # 创建测试用的.mmd文件内容
    test_content = """
# 化学反应机理研究

## 背景

在有机合成中，Diels-Alder反应是一个重要的环加成反应。该反应可以表示为：

$$\\ce{C4H6 + C2H4 -> C6H10}$$

更详细的机理如下：

$\\ce{diene + dienophile ->[heat] cyclohexene}$

## 实验部分

我们使用了以下化合物：
- 1,3-丁二烯 (C₄H₆)
- 马来酸酐 (C₄H₂O₃)
- 甲苯 (C₇H₈)

反应条件：
- 温度：180°C
- 压力：1 atm
- 时间：4 h

化学方程式：
C₄H₆ + C₄H₂O₃ → C₈H₈O₃

反应机理涉及协同过程，其中π电子重新分布。

## 结果与讨论

产物的NMR数据显示：
- ¹H NMR (CDCl₃): δ 7.2 (m, 2H), 3.1 (dd, 2H)
- ¹³C NMR: δ 171.2, 134.5, 52.3

动力学研究表明反应遵循二级动力学：
$$\\frac{d[P]}{dt} = k[A][B]$$

其中k = 2.3 × 10⁻⁴ M⁻¹s⁻¹ (at 180°C)

## 结论

该研究证明了Diels-Alder反应在温和条件下的有效性。
    """
    
    print("=== 化学公式处理器测试 ===\n")
    
    # 初始化处理器
    processor = ChemicalFormulaProcessor()
    
    # 处理文本
    print("原始文本：")
    print(test_content[:200] + "...\n")
    
    # 保护化学内容
    protected_text = processor.preserve_chemical_content(test_content)
    
    print("保护后的文本（部分）：")
    print(protected_text[:500] + "...\n")
    
    # 恢复化学内容
    restored_text = processor.restore_chemical_content(protected_text)
    
    print("恢复后的文本是否一致：", test_content == restored_text)
    
    # 详细分析
    print("\n=== 检测结果 ===")
    
    latex_formulas = processor.extract_latex_formulas(test_content)
    print(f"LaTeX公式 ({len(latex_formulas)}个):")
    for formula, start, end in latex_formulas:
        print(f"  - {formula}")
    
    chemical_entities = processor.detect_chemical_entities(test_content)
    print(f"\n化学实体 ({len(chemical_entities)}个):")
    for entity, start, end in chemical_entities[:10]:  # 只显示前10个
        print(f"  - {entity}")
    
    reaction_symbols = processor.detect_reaction_symbols(test_content)
    print(f"\n反应符号 ({len(reaction_symbols)}个):")
    for symbol, start, end in reaction_symbols:
        print(f"  - {symbol}")

def test_with_mmd_file(file_path: str):
    """测试真实的.mmd文件"""
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    print(f"\n=== 测试文件: {file_path} ===")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        processor = ChemicalFormulaProcessor()
        
        print(f"文件大小: {len(content)} 字符")
        
        # 检测化学内容
        latex_formulas = processor.extract_latex_formulas(content)
        chemical_entities = processor.detect_chemical_entities(content)
        reaction_symbols = processor.detect_reaction_symbols(content)
        
        print(f"检测结果:")
        print(f"  - LaTeX公式: {len(latex_formulas)}个")
        print(f"  - 化学实体: {len(chemical_entities)}个")
        print(f"  - 反应符号: {len(reaction_symbols)}个")
        
        if latex_formulas:
            print("\nLaTeX公式示例:")
            for formula, _, _ in latex_formulas[:3]:
                print(f"  {formula}")
        
        if chemical_entities:
            print("\n化学实体示例:")
            for entity, _, _ in chemical_entities[:10]:
                print(f"  {entity}")
        
        # 测试保护和恢复
        protected = processor.preserve_chemical_content(content)
        restored = processor.restore_chemical_content(protected)
        
        print(f"\n保护/恢复测试: {'通过' if content == restored else '失败'}")
        
    except Exception as e:
        print(f"测试文件时出错: {e}")

if __name__ == "__main__":
    # 运行基本测试
    test_chemical_formula_processor()
    
    # 测试真实文件
    mmd_dir = "./files_mmd"
    if os.path.exists(mmd_dir):
        mmd_files = [f for f in os.listdir(mmd_dir) if f.endswith('.mmd')]
        if mmd_files:
            # 测试第一个文件
            first_file = os.path.join(mmd_dir, mmd_files[0])
            test_with_mmd_file(first_file)
        else:
            print(f"\n在 {mmd_dir} 中未找到.mmd文件")
    else:
        print(f"\n目录不存在: {mmd_dir}")