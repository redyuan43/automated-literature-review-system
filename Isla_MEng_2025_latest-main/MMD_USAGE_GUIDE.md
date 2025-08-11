# MultiMarkdown (MMD) 文件使用指南

本指南说明如何使用通过 DotsOCR 公式保护系统生成的 MultiMarkdown 文件。

## 文件概述

通过 DotsOCR 处理的 MMD 文件具有以下特点：
- **完整公式恢复**：所有数学公式和化学反应式已从保护标签恢复为标准 LaTeX 格式
- **MultiMarkdown 格式**：包含元数据头，支持丰富的文档结构
- **学术论文友好**：保持原始文档的章节结构、公式编号和引用格式
- **多页面整合**：将原始 PDF 的多个页面合并为单一连贯文档

## 文件结构

### 元数据头
每个 MMD 文件开头包含元数据信息：
```markdown
Title: Document Title
Author: OCR Extracted Document  
Date: 2025-08-11
Format: MultiMarkdown
Language: en
Math: true
```

### 内容结构
- **文档标题**：主标题和处理说明
- **章节分层**：使用 `#`、`##` 等标记的标题层次
- **数学公式**：标准 LaTeX 格式，支持行内 `$...$` 和块级 `$$...$$` 公式
- **化学公式**：使用 mhchem 语法，如 `\ce{H2SO4}`
- **页面分隔**：使用 `---` 分隔原始 PDF 页面

## 外部工程集成

### 1. 基本文件处理

```python
# 读取 MMD 文件
def read_mmd_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

# 解析元数据
def parse_metadata(content):
    lines = content.split('\n')
    metadata = {}
    i = 0
    
    while i < len(lines) and lines[i].strip():
        line = lines[i].strip()
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
        i += 1
    
    # 返回元数据和正文内容
    document_content = '\n'.join(lines[i+1:])
    return metadata, document_content
```

### 2. LaTeX 公式渲染

MMD 文件中的公式可直接用于：

#### MathJax 渲染
```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async 
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                packages: {'[+]': ['mhchem']}
            }
        };
    </script>
</head>
<body>
    <!-- MMD 内容将被正确渲染 -->
</body>
</html>
```

#### Jupyter Notebook
```python
from IPython.display import Markdown, display

# 直接显示 MMD 内容
def display_mmd(file_path):
    content = read_mmd_file(file_path)
    display(Markdown(content))

# 提取特定部分
def extract_formulas(content):
    import re
    # 提取所有块级公式
    block_formulas = re.findall(r'\$\$(.+?)\$\$', content, re.DOTALL)
    # 提取所有行内公式  
    inline_formulas = re.findall(r'(?<!\$)\$([^$]+)\$(?!\$)', content)
    return block_formulas, inline_formulas
```

### 3. 转换为其他格式

#### 转换为标准 Markdown
```python
def mmd_to_markdown(mmd_content):
    lines = mmd_content.split('\n')
    
    # 跳过元数据头
    start_idx = 0
    while start_idx < len(lines):
        if lines[start_idx].strip() == '':
            start_idx += 1
            break
        if ':' not in lines[start_idx]:
            break
        start_idx += 1
    
    return '\n'.join(lines[start_idx:])
```

#### 转换为纯文本（保留公式）
```python
def extract_text_with_formulas(content):
    # 保留公式但移除其他 Markdown 标记
    import re
    
    # 保护公式
    protected_content = content
    formula_map = {}
    
    # 保护块级公式
    def protect_block_formula(match):
        key = f"__BLOCK_FORMULA_{len(formula_map)}__"
        formula_map[key] = match.group(0)
        return key
    
    protected_content = re.sub(r'\$\$.+?\$\$', protect_block_formula, 
                              protected_content, flags=re.DOTALL)
    
    # 保护行内公式
    def protect_inline_formula(match):
        key = f"__INLINE_FORMULA_{len(formula_map)}__"
        formula_map[key] = match.group(0)
        return key
    
    protected_content = re.sub(r'(?<!\$)\$([^$]+)\$(?!\$)', 
                              protect_inline_formula, protected_content)
    
    # 移除 Markdown 标记
    protected_content = re.sub(r'^#+\s*', '', protected_content, flags=re.MULTILINE)
    protected_content = re.sub(r'\*\*(.*?)\*\*', r'\1', protected_content)
    protected_content = re.sub(r'\*(.*?)\*', r'\1', protected_content)
    
    # 恢复公式
    for key, formula in formula_map.items():
        protected_content = protected_content.replace(key, formula)
    
    return protected_content
```

### 4. 搜索和信息提取

```python
class MMDProcessor:
    def __init__(self, mmd_file):
        self.content = read_mmd_file(mmd_file)
        self.metadata, self.document = parse_metadata(self.content)
    
    def search_formulas(self, pattern):
        """搜索包含特定模式的公式"""
        import re
        results = []
        
        # 搜索块级公式
        block_formulas = re.findall(r'\$\$(.+?)\$\$', self.document, re.DOTALL)
        for formula in block_formulas:
            if pattern in formula:
                results.append(('block', formula.strip()))
        
        # 搜索行内公式
        inline_formulas = re.findall(r'(?<!\$)\$([^$]+)\$(?!\$)', self.document)
        for formula in inline_formulas:
            if pattern in formula:
                results.append(('inline', formula.strip()))
        
        return results
    
    def extract_sections(self):
        """提取文档章节"""
        import re
        sections = {}
        lines = self.document.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            if re.match(r'^#+\s', line):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def get_formula_statistics(self):
        """获取公式统计信息"""
        import re
        
        block_count = len(re.findall(r'\$\$(.+?)\$\$', self.document, re.DOTALL))
        inline_count = len(re.findall(r'(?<!\$)\$([^$]+)\$(?!\$)', self.document))
        
        return {
            'block_formulas': block_count,
            'inline_formulas': inline_count,
            'total_formulas': block_count + inline_count
        }
```

## 使用示例

### 基本使用
```python
# 加载和显示文档
processor = MMDProcessor('document_complete.mmd')

print("文档元数据：")
print(processor.metadata)

print("\n公式统计：")
print(processor.get_formula_statistics())

print("\n搜索包含 'X_t' 的公式：")
results = processor.search_formulas('X_t')
for formula_type, formula in results:
    print(f"{formula_type}: {formula}")
```

### Web 应用集成
```python
# Flask 示例
from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/document/<doc_name>')
def show_document(doc_name):
    mmd_file = f"output/markdown/{doc_name}_complete.mmd"
    processor = MMDProcessor(mmd_file)
    
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ metadata.Title }}</title>
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    </head>
    <body>
        <h1>{{ metadata.Title }}</h1>
        <p>Author: {{ metadata.Author }}</p>
        <div>{{ document | safe }}</div>
    </body>
    </html>
    """
    
    return render_template_string(template, 
                                 metadata=processor.metadata,
                                 document=processor.document)
```

## 注意事项

1. **公式渲染**：确保前端支持 LaTeX 渲染（MathJax、KaTeX 等）
2. **化学公式**：需要 mhchem 扩展支持化学反应式渲染
3. **编码**：文件使用 UTF-8 编码，包含特殊数学符号
4. **分页**：原始页面用 `---` 分隔，可用于导航或分段处理
5. **元数据**：利用文件头的元数据进行分类和索引

## 输出目录结构

```
output/
├── markdown/
│   ├── document1_complete.mmd
│   ├── document2_complete.mmd
│   └── ...
└── protected/
    ├── document1/
    │   ├── page_1_protected.json
    │   ├── page_1_metadata.json
    │   └── ...
    └── ...
```

生成的 MMD 文件位于 `output/markdown/` 目录中，每个文档对应一个完整的 `.mmd` 文件。