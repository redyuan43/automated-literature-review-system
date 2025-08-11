import os
import json
from pathlib import Path


def export_reports(input_dir: str = "./outputs", output_dir: str = "./outputs/exports"):
    """从JSON报告文件生成Markdown导出"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 处理JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not json_files:
        print("❌ 未找到JSON文件")
        return

    for json_file in json_files:
        src = Path(input_dir) / json_file
        stem = src.stem
        
        # 读取JSON内容
        try:
            with open(src, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 提取内容
            title = data.get('chapter', '') + ' - ' + data.get('section_type', '')
            content = data.get('content', '')
            metadata = {
                'model': data.get('model', ''),
                'timestamp': data.get('timestamp', ''),
                'word_count': len(content.split())
            }
            
            # 生成Markdown内容
            md_content = f"""# {title}

**生成时间**: {metadata['timestamp']}  
**使用模型**: {metadata['model']}  
**字数统计**: {metadata['word_count']} 词

---

{content}

---
*由系统自动生成*
"""
            
        except Exception as e:
            print(f"❌ 处理JSON文件失败 {json_file}: {e}")
            continue

        # 保存 Markdown 文件
        dst_md = out_dir / f"{stem}.md"
        dst_md.write_text(md_content, encoding='utf-8')
        print(f"✅ 生成 Markdown: {dst_md}")

    print(f"\n🎉 导出完成！处理了 {len(json_files)} 个文件")
    print(f"📁 输出目录: {out_dir.absolute()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从JSON报告生成Markdown导出文件")
    parser.add_argument("--input-dir", "-i", default="./outputs", 
                       help="输入目录 (默认: ./outputs)")
    parser.add_argument("--output-dir", "-o", default="./outputs/exports",
                       help="输出目录 (默认: ./outputs/exports)")
    
    args = parser.parse_args()
    
    print(f"📂 输入目录: {args.input_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print()
    
    export_reports(args.input_dir, args.output_dir)


